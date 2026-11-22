"""The orchestrator: microphone → transcript → model → voice → robot, with interruption.

A turn is a three-stage pipeline running on its own threads:

    model text stream ─► sentences ─► TTS (PCM per sentence) ─► one paced HTTP stream to the
                                                                 speak server, firing gestures
                                                                 as each sentence is written

Pacing keeps the bytes written to the speak server only ~0.4 s ahead of what the robot is
actually saying, so a gesture fired "when the sentence is written" lands on the right words.
``interrupt()`` sets a cancel event that every stage checks, stops playback on the speak
server, and walks the robot's joints home through the daemon.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from companion import behaviour
from companion.audio import Microphone, VoiceGate
from companion.behaviour import State
from companion.config import Settings
from companion import memory as memstore
from companion.context import LiveContext
from companion.dialogue import Sentence, SentenceAssembler
from companion.llm import LLM, Conversation, Message, build_system_prompt, make_llm
from companion.choreography import (
    Planner, beat_hold, energy_bands, estimate_bpm, routines_for, script_for, song_routines, theme_for,
)
from companion.music import SAMPLE_RATE as SONG_RATE, Library, decode
from companion.wakeword import split_wake
from companion.robot import CarleProcesses, DaemonClient, DaemonError, SpeakClient
from companion.stt import STT, make_stt
from companion.tts import TTS, make_tts

_log = logging.getLogger(__name__)

Emit = Callable[[str, dict], None]

#: How far ahead of real time the speech writer may run (seconds of audio).
LEAD = 0.4
#: PCM write granularity (seconds).
CHUNK_SECONDS = 0.2
#: Keep the microphone muted this long after speech ends (room reverb, servo noise).
UNMUTE_DELAY = 0.35


@dataclass
class _Turn:
    id: int
    cancel: threading.Event = field(default_factory=threading.Event)
    spoken: list[str] = field(default_factory=list)  # sentences actually sent to the speaker
    raw: str = ""  # the model's full reply including tags


class Companion:
    def __init__(self, settings: Settings, secrets: dict[str, str], emit: Emit):
        self.settings = settings
        self.secrets = secrets
        self.emit = emit
        self.state = State.IDLE
        self.running = False
        self.processes: CarleProcesses | None = None
        self.daemon = DaemonClient()
        self.speak = SpeakClient(settings.speak_port)
        self.mic: Microphone | None = None
        self.gate: VoiceGate | None = None
        self.stt: STT | None = None
        self.tts: TTS | None = None
        self.llm: LLM | None = None
        self.conversation = Conversation(settings.max_history_turns)
        self._turn: _Turn | None = None
        self._turn_counter = 0
        self._turn_lock = threading.Lock()
        self._arms_busy_until = 0.0
        self._face = behaviour.FACE_CLEAR
        self._poller: threading.Thread | None = None
        self._stop_poll = threading.Event()
        self.last_status: dict = {}
        self.provider_errors: dict[str, str] = {}
        self.library = Library(settings.music_dir)
        self.context = LiveContext(settings.location, owner=settings.owner_name, birthday=settings.owner_birthday, birthdays=settings.birthdays)
        self.memories: list[str] = memstore.load() if settings.memory_enabled else []
        self._memory_llm: LLM | None = None
        self._turns_since_memory = 0
        self._memory_lock = threading.Lock()
        self._party_cancel = threading.Event()
        self._wake_until = 0.0
        self._dance_cancel = threading.Event()
        self._last_interaction = time.monotonic()
        self._nudges = 0
        self._greeted = False
        self._disconnected_since: float | None = None
        self._tts_ready = False

    # --- lifecycle ------------------------------------------------------------------------

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        s = self.settings
        self.processes = CarleProcesses(s.robot_address, s.speak_port, s.speaker_device)
        self.processes.start()
        self._build_providers()
        threading.Thread(target=self._open_microphone, name="mic-open", daemon=True).start()
        self._stop_poll.clear()
        self._poller = threading.Thread(target=self._poll_status, name="status-poller", daemon=True)
        self._poller.start()
        threading.Thread(target=self._warm_up, name="warm-up", daemon=True).start()
        threading.Thread(target=self._refresh_context, name="context", daemon=True).start()
        self._set_state(State.LISTENING if s.hands_free else State.IDLE)

    def _open_microphone(self) -> None:
        """Ask macOS for microphone access (shows the prompt if undecided), then open the mic."""
        from companion.permissions import request_microphone  # noqa: PLC0415

        s = self.settings
        status = request_microphone(timeout=180)
        if status != "authorized":
            self.provider_errors["mic"] = f"microphone permission {status}"
            self.emit("error", {
                "message": f"Microphone permission is {status}. Allow RukoCompanion under System "
                "Settings → Privacy & Security → Microphone, then press Stop and Start."
            })
            if status != "not-determined":
                return
        if not self.running:
            return
        try:
            self.mic = Microphone(s.mic_device, on_level=lambda lvl: self.emit("level", {"rms": lvl}))
            self.mic.open()
            self.provider_errors.pop("mic", None)
        except Exception as exc:  # noqa: BLE001 - report, keep running without a mic
            self.mic = None
            self.provider_errors["mic"] = str(exc)
            self.emit("error", {"message": f"Microphone: {exc}"})
            return
        self.gate = VoiceGate(self.mic, self._on_hands_free_utterance, threshold=s.vad_threshold)
        if s.hands_free:
            self.gate.enable()
        self.emit("status", self.status())

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.interrupt(reason="shutdown")
        if self._turns_since_memory:
            self._remember()  # a last pass before the lights go out (bounded by the model's reply)
        self._stop_poll.set()
        if self.gate:
            self.gate.disable()
        if self.mic:
            self.mic.close()
        try:
            self.daemon.enqueue(behaviour.home_items())
            time.sleep(1.5)
        except DaemonError:
            pass
        if self.processes:
            self.processes.stop()
        if hasattr(self.llm, "close"):
            self.llm.close()
        self._set_state(State.IDLE, announce_only=True)

    def apply_settings(self, settings: Settings, secrets: dict[str, str]) -> None:
        """Hot-apply a settings change (providers, personality, robot address, hands-free)."""
        old = self.settings
        self.settings = settings
        self.secrets = secrets
        self.conversation.max_turns = settings.max_history_turns
        if settings.music_dir != old.music_dir:
            self.library = Library(settings.music_dir)
        if settings.location != old.location:
            self.context.set_place(settings.location)
            threading.Thread(target=self._refresh_context, daemon=True).start()
        self.context.owner, self.context.birthday, self.context.birthdays = settings.owner_name, settings.owner_birthday, settings.birthdays
        if settings.memory_enabled and not old.memory_enabled:
            self.memories = memstore.load()
        if self.running:
            if self.processes and settings.robot_address != old.robot_address:
                self.processes.set_address(settings.robot_address)
            if (
                settings.stt_provider, settings.stt_model, settings.tts_provider, settings.tts_voice,
                settings.tts_speed, settings.llm_provider, settings.llm_model, settings.llm_base_url,
                settings.max_reply_tokens,
            ) != (
                old.stt_provider, old.stt_model, old.tts_provider, old.tts_voice, old.tts_speed,
                old.llm_provider, old.llm_model, old.llm_base_url, old.max_reply_tokens,
            ) or secrets != self.secrets:
                self._build_providers()
                threading.Thread(target=self._warm_up, daemon=True).start()
            if self.gate and settings.hands_free != old.hands_free:
                self.set_hands_free(settings.hands_free)
            if self.mic and settings.mic_device != old.mic_device:
                self.mic.close()
                self.mic = Microphone(settings.mic_device, on_level=lambda lvl: self.emit("level", {"rms": lvl}))
                self.mic.open()
                self.gate = VoiceGate(self.mic, self._on_hands_free_utterance, threshold=settings.vad_threshold)
                if settings.hands_free:
                    self.gate.enable()
        self.emit("status", self.status())

    def _build_providers(self) -> None:
        self.provider_errors = {k: v for k, v in self.provider_errors.items() if k == "mic"}
        if hasattr(self.llm, "close"):
            self.llm.close()  # a persistent Claude Code process, if one is running
        for name, factory in (("stt", make_stt), ("tts", make_tts), ("llm", make_llm)):
            try:
                setattr(self, name, factory(self.settings, self.secrets))
            except Exception as exc:  # noqa: BLE001 - surface in the UI, don't crash
                setattr(self, name, None)
                self.provider_errors[name] = str(exc)
                _log.warning("%s provider unavailable: %s", name, exc)
                self.emit("error", {"message": str(exc)})

    def _refresh_context(self) -> None:
        if not self.settings.live_context:
            return
        try:
            self.context.refresh()
            if self.context.resolved_place and not self.settings.location:
                self.emit("notice", {"message": f"Weather location detected: {self.context.resolved_place} (change it in Settings)"})
        except Exception as exc:  # noqa: BLE001
            _log.debug("context refresh failed: %s", exc)

    def _warm_up(self) -> None:
        for p in (self.stt, self.tts):
            if p is None:
                continue
            try:
                p.warm_up()
                if p is self.tts:
                    self._tts_ready = True
            except Exception as exc:  # noqa: BLE001
                self.provider_errors[p.name] = str(exc)
                self.emit("error", {"message": f"{p.name}: {exc}"})
        self.emit("status", self.status())

    # --- status ---------------------------------------------------------------------------

    def status(self) -> dict:
        robot = {"connected": False, "queued": 0, "doing": None}
        try:
            st = self.daemon.status()
            robot = {
                "connected": bool(st.get("connected")),
                "queued": st.get("queued", 0),
                "doing": st.get("current"),
            }
        except DaemonError:
            pass
        info = {
            "running": self.running,
            "state": self.state.value,
            "robot": robot,
            "daemon_process": bool(self.processes and self.processes.daemon_alive),
            "speaker": self.speak.alive() if self.running else False,
            "speak_process": bool(self.processes and self.processes.speak_alive),
            "mic": bool(self.mic and self.mic._stream is not None),
            "mic_name": self.settings.mic_device,
            "hands_free": bool(self.gate and self.gate.enabled),
            "providers": {
                "stt": getattr(self.stt, "name", None),
                "tts": getattr(self.tts, "name", None),
                "llm": getattr(self.llm, "name", None),
            },
            "provider_errors": dict(self.provider_errors),
            "restarts": dict(self.processes.restarts) if self.processes else {},
            "robot_address": self.settings.robot_address,
        }
        self.last_status = info
        return info

    def _poll_status(self) -> None:
        from companion.speaker import SpeakerKeeper  # noqa: PLC0415

        keeper = SpeakerKeeper(self.settings.speaker_device)
        was_connected: bool | None = None
        down_since: float | None = None
        ticks = 0
        while not self._stop_poll.wait(2.0):
            ticks += 1
            if ticks % 150 == 0:  # every five minutes, cheap because the context caches
                threading.Thread(target=self._refresh_context, daemon=True).start()
            info = self.status()
            # A daemon that stays disconnected for a long while is usually stuck with stale
            # CoreBluetooth state (Mac Bluetooth toggled, radio reset): relaunch it.
            if info["robot"]["connected"] or not info["daemon_process"]:
                down_since = None
            elif down_since is None:
                down_since = time.monotonic()
            elif time.monotonic() - down_since > 60 and self.processes:
                self.processes.restart_daemon("robot link down for over 60 s")
                self.emit("notice", {"message": "Robot link down for a minute — restarting the Bluetooth link"})
                down_since = None
            # If the robot is reachable over BLE but its speaker dropped, ask macOS to
            # reconnect it (rate-limited inside the keeper).
            info["speaker_bt"] = keeper.ensure() if info["robot"]["connected"] else keeper.ensure() if info["speaker"] is False else None
            self.emit("status", info)
            connected = info["robot"]["connected"]
            if was_connected is not None and connected != was_connected:
                self.emit("notice", {"message": "Robot connected" if connected else "Robot link lost — reconnecting"})
                if connected and self.state != State.IDLE:
                    self._apply_face(behaviour.STATE_FACES[self.state], force=True)
            if not connected:
                if self._disconnected_since is None:
                    self._disconnected_since = time.monotonic()
            else:
                # Greet when the robot first comes online, and again if it was away long enough
                # to have been switched off (it probably just came back).
                away = (time.monotonic() - self._disconnected_since) if self._disconnected_since else 0.0
                self._disconnected_since = None
                if (not self._greeted or away > 90) and self._ready_to_talk(info):
                    self._greeted = True
                    self._initiate("greet")
            was_connected = connected
            self._maybe_nudge(info)

    # --- conversation initiative -------------------------------------------------------------

    def _ready_to_talk(self, info: dict) -> bool:
        return bool(
            self.running and self.settings.auto_greet and self._tts_ready and self.llm is not None
            and info.get("speaker") and info["robot"]["connected"] and not self.busy
        )

    def _initiate(self, kind: str) -> None:
        """Have the robot speak first: a greeting or an idle nudge. Runs as a normal turn."""
        owner = self.settings.owner_name or "everyone"
        if kind == "greet":
            prompt = (
                f"(You have just been switched on and {owner} may be nearby. Say a warm, short "
                f"greeting of one or two sentences and suggest one fun thing you could do together.)"
            )
        else:
            prompt = (
                f"(Nobody has said anything for a while. In one or two short sentences, say "
                f"something to get {owner}'s attention: offer a dance, a dinosaur fact, a chicken "
                f"noise, or ask a simple question. Do not repeat what you said last time.)"
            )
        _log.info("initiating conversation: %s", kind)
        threading.Thread(target=self._run_turn, args=(prompt, False), daemon=True).start()

    def _maybe_nudge(self, info: dict) -> None:
        minutes = self.settings.idle_nudge_minutes
        if minutes <= 0 or not self.settings.hands_free or not self._ready_to_talk(info):
            return
        if self._nudges >= self.settings.idle_nudge_max:
            return
        if time.monotonic() - self._last_interaction >= minutes * 60:
            self._nudges += 1
            self._last_interaction = time.monotonic()
            self._initiate("nudge")

    def _open_follow_up_window(self) -> None:
        """After the robot speaks, accept the next utterance without the wake word."""
        if self.settings.hands_free and self.settings.follow_up_seconds > 0:
            self._wake_until = time.monotonic() + self.settings.follow_up_seconds

    def _note_interaction(self) -> None:
        self._last_interaction = time.monotonic()
        self._nudges = 0

    # --- user actions ---------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        """True while a reply is being generated or spoken, or music is playing."""
        return self.state in (State.THINKING, State.SPEAKING, State.DANCING)

    # --- memory across days -------------------------------------------------------------------

    def _remember(self) -> None:
        """Distil recent turns into durable facts; runs in the background every few turns."""
        if not self.settings.memory_enabled or self.llm is None:
            return
        if not self._memory_lock.acquire(blocking=False):
            return
        try:
            self._turns_since_memory = 0
            if self._memory_llm is None:
                self._memory_llm = make_llm(self.settings, self.secrets)  # its own session
            recent = self.conversation.messages[-12:]
            if len(recent) < 2:
                return
            new = memstore.extract(self._memory_llm, self.settings.owner_name, recent, self.memories)
            if new:
                before = len(self.memories)
                self.memories = memstore.add(self.memories, new)
                if len(self.memories) >= memstore.MAX_LINES:
                    self.memories = memstore.merge(self._memory_llm, self.memories)
                memstore.save(self.memories)
                added = self.memories[before:] if len(self.memories) > before else new
                self.emit("memory", {"added": added, "count": len(self.memories)})
                _log.info("remembered %d new fact(s)", len(added))
        except Exception as exc:  # noqa: BLE001 - memory must never break the conversation
            _log.warning("memory extraction failed: %s", exc)
        finally:
            self._memory_lock.release()

    def save_memories(self, lines: list[str]) -> None:
        self.memories = [l.strip().lstrip("- ").strip() for l in lines if l.strip()]
        memstore.save(self.memories)

    def song_titles(self) -> list[str]:
        self.library.rescan()
        return self.library.titles()

    def play_song(self, title: str) -> bool:
        """Play a song from the library through the robot and dance to it."""
        if not self.running:
            return False
        if title.strip().lower() in ("party", "all", "everything", "dance party"):
            if self.busy:
                self.interrupt(reason="dance party")
            threading.Thread(target=self._dance_party, daemon=True).start()
            return True
        if self.library.find(title) is None:
            return False
        if self.busy:
            self.interrupt(reason="new song")
        threading.Thread(target=self._dance, args=(title,), daemon=True).start()
        return True

    def _dance_party(self) -> None:
        """Every song in the library, shuffled, until someone interrupts."""
        import random  # noqa: PLC0415

        self._party_cancel = threading.Event()
        cancel = self._party_cancel
        songs = list(self.library.songs)
        random.shuffle(songs)
        self.emit("notice", {"message": f"Dance party: {len(songs)} songs"})
        for song in songs:
            if cancel.is_set() or not self.running:
                break
            self._dance(song.title)
            time.sleep(1.0)

    def _pronounce(self, text: str) -> str:
        for word, spoken in self.settings.pronunciation_map().items():
            text = re.sub(r"\b" + re.escape(word) + r"\b", spoken, text)
        return text

    def press_talk(self) -> None:
        if not self.running or not self.mic:
            return
        if self.busy:
            self.interrupt(reason="user pressed talk")
        self.mic.set_muted(False)
        self.mic.start_recording()
        self._set_state(State.LISTENING)

    def release_talk(self) -> None:
        if not self.running or not self.mic:
            return
        audio = self.mic.stop_recording()
        if len(audio) < 0.3 * 16000:
            self.emit("notice", {"message": "Too short — hold the button while you speak"})
            self._set_state(State.LISTENING if self.settings.hands_free else State.IDLE)
            return
        threading.Thread(target=self._handle_utterance, args=(audio,), daemon=True).start()

    def send_text(self, text: str) -> None:
        """Typed input: skips the microphone and speech-to-text."""
        text = text.strip()
        if not text or not self.running:
            return
        if self.busy:
            self.interrupt(reason="new typed message")
        self._note_interaction()
        self.emit("transcript", {"role": "user", "text": text})
        threading.Thread(target=self._run_turn, args=(text,), daemon=True).start()

    def set_hands_free(self, enabled: bool) -> None:
        self.settings.hands_free = enabled
        if not self.gate:
            return
        if enabled:
            self.gate.enable()
            if self.state == State.IDLE:
                self._set_state(State.LISTENING)
        else:
            self.gate.disable()
            if self.state == State.LISTENING:
                self._set_state(State.IDLE)
        self.emit("status", self.status())

    def interrupt(self, reason: str = "interrupted") -> None:
        turn = self._turn
        if turn:
            turn.cancel.set()
        self._dance_cancel.set()
        self._party_cancel.set()
        self.speak.stop()
        try:
            self.daemon.clear()
            self.daemon.stop()  # Carle walks the joints home and releases the face
        except DaemonError:
            pass
        self._face = behaviour.FACE_CLEAR
        self._arms_busy_until = 0.0
        if self.mic:
            self.mic.set_muted(False)
        _log.info("interrupted: %s", reason)

    def emergency_stop(self) -> None:
        """Stop everything that moves or makes sound, now."""
        self.interrupt(reason="EMERGENCY STOP")
        if self.mic:
            self.mic.stop_recording()
        if self.gate:
            self.gate.disable()
            self.settings.hands_free = False
        self._set_state(State.IDLE)
        self.emit("notice", {"message": "Emergency stop: motion and audio halted"})

    def reset_conversation(self) -> None:
        if self._turns_since_memory:
            threading.Thread(target=self._remember, daemon=True).start()
        self.conversation.reset()
        if hasattr(self.llm, "new_session"):
            self.llm.new_session()  # Claude Code keeps its own memory per session
        self.emit("transcript_reset", {})

    # --- the turn -------------------------------------------------------------------------

    def _on_hands_free_utterance(self, audio: np.ndarray) -> None:
        if self.state in (State.SPEAKING, State.THINKING, State.DANCING):
            # Barge-in: the user spoke while we were busy (mic is muted during speech, so this
            # only fires in the thinking gap) — treat it as an interruption.
            self.interrupt(reason="barge-in")
        threading.Thread(target=self._handle_utterance, args=(audio, True), daemon=True).start()

    def _handle_utterance(self, audio: np.ndarray, hands_free: bool = False) -> None:
        if self.stt is None:
            self.emit("error", {"message": "No speech-to-text provider is configured"})
            return
        if len(audio) and not np.any(audio):
            # macOS delivers pure digital silence when the app has no microphone permission.
            self.provider_errors["mic"] = "no microphone permission"
            self.emit("error", {
                "message": "The microphone is delivering silence: allow RukoCompanion under "
                "System Settings → Privacy & Security → Microphone, then press Stop and Start."
            })
            self._set_state(State.LISTENING if self.settings.hands_free else State.IDLE)
            return
        self._set_state(State.THINKING)
        t0 = time.monotonic()
        try:
            text = self.stt.transcribe(audio, 16000)
        except Exception as exc:  # noqa: BLE001
            self.emit("error", {"message": f"Speech recognition failed: {exc}"})
            self._set_state(State.LISTENING if self.settings.hands_free else State.IDLE)
            return
        _log.info("stt %.2fs: %r", time.monotonic() - t0, text)
        if not text or len(text.strip(" .")) < 2:
            self.emit("notice", {"message": "I didn't catch that"})
            self._set_state(State.LISTENING if self.settings.hands_free else State.IDLE)
            return
        if hands_free and self.settings.wake_word_enabled:
            matched, rest = split_wake(text, self.settings.wake_word)
            window_open = time.monotonic() < self._wake_until
            if not matched and not window_open:
                _log.info("ignored, no wake word: %r", text)
                self._set_state(State.LISTENING)
                return
            if matched and not rest:
                # Just the name: acknowledge and take the next utterance as the question.
                self._wake_until = time.monotonic() + self.settings.wake_window_seconds
                self.emit("wake", {"message": f"{self.settings.wake_word} is listening…"})
                self._say_short("Yes?")
                self._set_state(State.LISTENING)
                return
            text = rest if matched else text
            self._wake_until = 0.0
        self._note_interaction()
        self.emit("transcript", {"role": "user", "text": text})
        self._run_turn(text)

    def _run_turn(self, user_text: str, show_user: bool = True) -> None:
        if self.llm is None or self.tts is None:
            self.emit("error", {"message": "A language model and a voice must be configured first"})
            self._set_state(State.LISTENING if self.settings.hands_free else State.IDLE)
            return
        with self._turn_lock:
            self._turn_counter += 1
            turn = _Turn(self._turn_counter)
            self._turn = turn
        self.conversation.add_user(user_text)
        self._set_state(State.THINKING)
        if self.mic:
            self.mic.set_muted(True)

        sentences: queue.Queue[Sentence | None] = queue.Queue()
        pcm_q: queue.Queue[tuple[Sentence, np.ndarray] | None] = queue.Queue(maxsize=3)
        t_start = time.monotonic()

        def produce_text() -> None:
            assembler = SentenceAssembler()
            try:
                messages = self.conversation.messages
                if self.settings.live_context and messages:
                    # Time and weather ride along with the question, not in the system prompt,
                    # so the warm Claude Code process never has to restart for a new minute.
                    last = messages[-1]
                    messages = messages[:-1] + [Message(last.role, f"({self.context.text()})\n{last.content}")]
                system = build_system_prompt(self.settings, self.library.titles(), memstore.for_prompt(self.memories))
                for chunk in self.llm.stream(system, messages):
                    if turn.cancel.is_set():
                        break
                    turn.raw += chunk
                    for s in assembler.feed(chunk):
                        sentences.put(s)
                if not turn.cancel.is_set():
                    for s in assembler.flush():
                        sentences.put(s)
            except Exception as exc:  # noqa: BLE001
                _log.exception("model error")
                self.emit("error", {"message": f"Model: {exc}"})
                sentences.put(Sentence("Sorry, I couldn't reach my language model just now.", ["sad"]))
            finally:
                sentences.put(None)

        def synthesize() -> None:
            try:
                while True:
                    s = sentences.get()
                    if s is None or turn.cancel.is_set():
                        break
                    if not s.speakable:
                        pcm = np.zeros(0, dtype=np.int16)
                    else:
                        t = time.monotonic()
                        try:
                            pcm = self.tts.synthesize(self._pronounce(s.text))
                        except Exception as exc:  # noqa: BLE001
                            _log.exception("tts error")
                            self.emit("error", {"message": f"Voice: {exc}"})
                            continue
                        _log.debug("tts %.2fs for %d chars", time.monotonic() - t, len(s.text))
                    while not turn.cancel.is_set():
                        try:
                            pcm_q.put((s, pcm), timeout=0.2)
                            break
                        except queue.Full:
                            continue
            finally:
                while True:
                    try:
                        pcm_q.put(None, timeout=0.2)
                        break
                    except queue.Full:
                        if turn.cancel.is_set():
                            break

        threading.Thread(target=produce_text, name=f"llm-{turn.id}", daemon=True).start()
        threading.Thread(target=synthesize, name=f"tts-{turn.id}", daemon=True).start()

        # The speaking stage runs here: the HTTP body generator pulls from pcm_q.
        first_audio_at: float | None = None
        requested_actions: list[tuple[str, str]] = []

        def body(first_item):
            nonlocal first_audio_at
            start: float | None = None
            written = 0.0
            sr = self.tts.samplerate
            chunk = int(sr * CHUNK_SECONDS)
            item = first_item
            while True:
                if item is None or turn.cancel.is_set():
                    return
                s, pcm = item
                if start is None:
                    start = time.monotonic()
                    first_audio_at = start
                    self._set_state(State.SPEAKING)
                # pace: never run more than LEAD seconds ahead of the robot's mouth
                while not turn.cancel.is_set():
                    ahead = written - (time.monotonic() - start)
                    if ahead <= LEAD:
                        break
                    time.sleep(min(ahead - LEAD, 0.1))
                if turn.cancel.is_set():
                    return
                self._express(s)
                requested_actions.extend(s.actions)
                if s.speakable:
                    turn.spoken.append(s.text)
                    self.emit("transcript", {"role": "assistant", "text": " ".join(turn.spoken), "partial": True})
                for i in range(0, len(pcm), chunk):
                    if turn.cancel.is_set():
                        return
                    yield pcm[i : i + chunk].tobytes()
                written += len(pcm) / sr
                item = pcm_q.get()

        outcome = "no-audio"
        unvoiced_text = ""
        try:
            # Wait for the first sentence before opening the HTTP stream so a cancelled or
            # empty reply never opens a playback.
            first = pcm_q.get()
            if first is not None and not turn.cancel.is_set():
                result = self.speak.stream_pcm(body(first), self.tts.samplerate)
                outcome = result.get("outcome", result.get("error", "unknown"))
                if not result.get("ok"):
                    self.emit("error", {"message": f"Speaker: {result.get('error')}"})
                    if not turn.spoken:
                        # Nothing was voiced: still show what the robot would have said.
                        from companion.dialogue import strip_tags  # noqa: PLC0415

                        time.sleep(0.5)  # let the model finish so the text is complete
                        text, _ = strip_tags(turn.raw)
                        if text:
                            unvoiced_text = text
        except Exception as exc:  # noqa: BLE001
            _log.exception("speak error")
            self.emit("error", {"message": f"Speaker: {exc}"})
        finally:
            _log.info(
                "turn %d: %s; first audio after %.2fs; total %.1fs",
                turn.id, outcome,
                (first_audio_at - t_start) if first_audio_at else -1,
                time.monotonic() - t_start,
            )
            interrupted = turn.cancel.is_set() and self.running
            said = " ".join(turn.spoken)
            if turn.spoken and interrupted:
                said += " …(interrupted)"
            if turn.raw.strip():
                # What the model sees later: its full reply, marked if it was cut off.
                self.conversation.add_assistant(
                    turn.raw.strip() + ("\n(You were interrupted before finishing.)" if interrupted else "")
                )
            if turn.spoken:
                self.emit("transcript", {"role": "assistant", "text": said, "partial": False})
            elif unvoiced_text:
                self.emit("transcript", {"role": "assistant", "text": unvoiced_text + " (robot speaker unavailable)", "partial": False})
            if self._turn is turn:
                if not interrupted and turn.raw.strip():
                    self._turns_since_memory += 1
                    if self._turns_since_memory >= 4:
                        threading.Thread(target=self._remember, name="remember", daemon=True).start()
                dance = next((arg for name, arg in requested_actions if name == "dance"), None)
                if dance and not interrupted and self.running:
                    if dance.strip().lower() in ("party", "all", "everything", "dance party"):
                        self._dance_party()
                    else:
                        self._dance(dance)  # keeps the mic muted and hands back when done
                    return
                time.sleep(UNMUTE_DELAY)
                if self.mic:
                    self.mic.set_muted(False)
                self._open_follow_up_window()
                self._set_state(State.LISTENING if self.settings.hands_free else State.IDLE)

    # --- music ------------------------------------------------------------------------------

    def _say_short(self, text: str) -> None:
        """Voice a short phrase without the model (acknowledgements)."""
        if self.tts is None:
            return
        try:
            pcm = self.tts.synthesize(self._pronounce(text))
        except Exception as exc:  # noqa: BLE001
            self.emit("error", {"message": f"Voice: {exc}"})
            return
        if self.mic:
            self.mic.set_muted(True)
        self._set_state(State.SPEAKING)
        try:
            self.speak.stream_pcm(iter([pcm.tobytes()]), self.tts.samplerate)
        except Exception as exc:  # noqa: BLE001
            _log.debug("short say failed: %s", exc)
        finally:
            time.sleep(0.2)
            if self.mic:
                self.mic.set_muted(False)

    def _dance(self, title: str) -> None:
        song = self.library.find(title)
        if song is None:
            self.emit("notice", {"message": f"No song matching '{title}'"})
            return
        cancel = threading.Event()
        self._dance_cancel = cancel
        try:
            pcm = decode(song)
        except Exception as exc:  # noqa: BLE001
            self.emit("error", {"message": f"Could not decode {song.title}: {exc}"})
            return
        if self.mic:
            self.mic.set_muted(True)
        self._set_state(State.DANCING)
        self.emit("transcript", {"role": "assistant", "text": f"♪ {song.title}", "partial": False})
        sr = SONG_RATE
        chunk = int(sr * CHUNK_SECONDS)
        bpm = estimate_bpm(pcm, sr)
        hold = beat_hold(bpm)
        theme = theme_for(song.title)
        script = script_for(song.title)
        registry = song_routines(hold)
        floor_ok = self.settings.dance_floor_moves
        planner = Planner(routines_for(theme, hold, floor_ok))
        total = len(pcm) / sr
        band_window = 2.0
        bands = energy_bands(pcm, sr, band_window)
        state = {"next": 0.0, "face_at": 0.0, "face_i": 0, "band": -1, "routines": [], "section": None}
        if script:
            self.emit("notice", {"message": f"Bespoke choreography for {script.title} at about {bpm:.0f} beats per minute"})
        else:
            self.emit("notice", {"message": f"{theme.description.capitalize()} at about {bpm:.0f} beats per minute"})

        def pick_routine(position: float):
            """Scripted: the section's signature on entry, then its pool; else the theme planner."""
            if script is None:
                return planner.next(position / total if total else 0.0)
            section = script.section_at(position)
            if section is not state["section"]:
                state["section"], state["face_i"], state["face_at"] = section, 0, 0.0
                if section.note:
                    self.emit("gesture", {"tag": f"section: {section.note}"})
                if section.signature and section.signature in registry:
                    return registry[section.signature]
            pool = [registry[n] for n in section.pool if n in registry and (floor_ok or not registry[n].floor)]
            pool = [r for r in pool if r.name != (state["routines"][-1] if state["routines"] else None)] or pool
            return pool[planner.count % len(pool)] if pool else planner.next(position / total)

        def body():
            start = time.monotonic()
            written = 0.0
            for i in range(0, len(pcm), chunk):
                if cancel.is_set():
                    return
                while not cancel.is_set():
                    ahead = written - (time.monotonic() - start)
                    if ahead <= LEAD:
                        break
                    time.sleep(min(ahead - LEAD, 0.1))
                if cancel.is_set():
                    return
                now = time.monotonic()
                if now >= state["next"]:
                    routine = pick_routine(written)
                    planner.count += 1
                    try:
                        self.daemon.enqueue(list(routine.steps))
                        state["routines"].append(routine.name)
                        self.emit("gesture", {"tag": f"dance: {routine.name}"})
                    except DaemonError:
                        pass
                    # Next routine starts when this one has finished, plus one beat's rest.
                    state["next"] = now + routine.duration + hold
                # The face follows the music: quiet/medium/loud passages each have their own
                # faces for this theme, cycled every face_period seconds.
                band = int(bands[min(len(bands) - 1, int(written / band_window))])
                if script is not None:
                    section = script.section_at(written)
                    if now >= state["face_at"]:
                        palette = section.faces
                        self._apply_face(palette[state["face_i"] % len(palette)])
                        state["face_i"] += 1
                        state["face_at"] = now + script.face_period
                elif band != state["band"] or now >= state["face_at"]:
                    if band != state["band"]:
                        state["band"], state["face_i"] = band, 0
                    else:
                        state["face_i"] += 1
                    palette = theme.faces[band]
                    self._apply_face(palette[state["face_i"] % len(palette)])
                    state["face_at"] = now + theme.face_period
                yield pcm[i : i + chunk].tobytes()
                written += len(pcm[i : i + chunk]) / sr

        _log.info("dancing to %s (%s theme, %.0fs, %.0f bpm, hold %.2fs)", song.title, theme.name, total, bpm, hold)
        try:
            result = self.speak.stream_pcm(body(), sr)
            if not result.get("ok"):
                self.emit("error", {"message": f"Speaker: {result.get('error')}"})
        except Exception as exc:  # noqa: BLE001
            self.emit("error", {"message": f"Speaker: {exc}"})
        finally:
            try:
                self.daemon.clear()
                self.daemon.enqueue(behaviour.home_items())
            except DaemonError:
                pass
            self._arms_busy_until = 0.0
            self._face = behaviour.FACE_CLEAR
            time.sleep(UNMUTE_DELAY)
            if self.mic:
                self.mic.set_muted(False)
            self._open_follow_up_window()
            if self.running:
                self._set_state(State.LISTENING if self.settings.hands_free else State.IDLE)

    # --- robot expression -----------------------------------------------------------------

    def _set_state(self, state: State, announce_only: bool = False) -> None:
        self.state = state
        self.emit("state", {"state": state.value})
        if announce_only:
            return
        if state == State.IDLE:
            self._apply_face(behaviour.FACE_CLEAR)
        else:
            self._apply_face(behaviour.STATE_FACES[state])

    def _apply_face(self, code: int, force: bool = False) -> None:
        if code == self._face and not force:
            return
        if not self.last_status.get("robot", {}).get("connected"):
            return  # don't pile faces into the queue while the link is down
        try:
            self.daemon.enqueue([behaviour.face_item(code)])
            self._face = code
        except DaemonError:
            pass

    def _express(self, s: Sentence) -> None:
        """Fire a sentence's approved gestures and emotional face, if the arms are free."""
        face = behaviour.face_for(s.tags, behaviour.FACE_TALKING)
        self._apply_face(face)
        now = time.monotonic()
        for tag in s.tags:
            g = behaviour.GESTURES.get(tag)
            if not g or not g.moves:
                continue
            if now < self._arms_busy_until:
                _log.debug("skipping [%s]: arms busy", tag)
                continue
            try:
                self.daemon.enqueue(g.steps)
                self._arms_busy_until = now + g.duration + 0.3
                self.emit("gesture", {"tag": tag})
            except DaemonError as exc:
                _log.debug("gesture [%s] dropped: %s", tag, exc)
            break  # one arm gesture per sentence is plenty
