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
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from companion import behaviour
from companion.audio import Microphone, VoiceGate
from companion.behaviour import State
from companion.config import Settings
from companion.dialogue import Sentence, SentenceAssembler
from companion.llm import LLM, Conversation, build_system_prompt, make_llm
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

    # --- lifecycle ------------------------------------------------------------------------

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        s = self.settings
        self.processes = CarleProcesses(s.robot_address, s.speak_port, s.speaker_device)
        self.processes.start()
        self._build_providers()
        try:
            self.mic = Microphone(s.mic_device, on_level=lambda lvl: self.emit("level", {"rms": lvl}))
            self.mic.open()
        except Exception as exc:  # noqa: BLE001 - report, keep running without a mic
            self.mic = None
            self.provider_errors["mic"] = str(exc)
            self.emit("error", {"message": f"Microphone: {exc}"})
        if self.mic:
            self.gate = VoiceGate(self.mic, self._on_hands_free_utterance, threshold=s.vad_threshold)
            if s.hands_free:
                self.gate.enable()
        self._stop_poll.clear()
        self._poller = threading.Thread(target=self._poll_status, name="status-poller", daemon=True)
        self._poller.start()
        threading.Thread(target=self._warm_up, name="warm-up", daemon=True).start()
        self._set_state(State.LISTENING if s.hands_free else State.IDLE)

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.interrupt(reason="shutdown")
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

    def _warm_up(self) -> None:
        for p in (self.stt, self.tts):
            if p is None:
                continue
            try:
                p.warm_up()
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
        was_connected: bool | None = None
        while not self._stop_poll.wait(2.0):
            info = self.status()
            self.emit("status", info)
            connected = info["robot"]["connected"]
            if was_connected is not None and connected != was_connected:
                self.emit("notice", {"message": "Robot connected" if connected else "Robot link lost — reconnecting"})
                if connected and self.state != State.IDLE:
                    self._apply_face(behaviour.STATE_FACES[self.state], force=True)
            was_connected = connected

    # --- user actions ---------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        """True while a reply is being generated or spoken."""
        return self.state in (State.THINKING, State.SPEAKING)

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
        self.conversation.reset()
        if hasattr(self.llm, "new_session"):
            self.llm.new_session()  # Claude Code keeps its own memory per session
        self.emit("transcript_reset", {})

    # --- the turn -------------------------------------------------------------------------

    def _on_hands_free_utterance(self, audio: np.ndarray) -> None:
        if self.state in (State.SPEAKING, State.THINKING):
            # Barge-in: the user spoke while we were busy (mic is muted during speech, so this
            # only fires in the thinking gap) — treat it as an interruption.
            self.interrupt(reason="barge-in")
        threading.Thread(target=self._handle_utterance, args=(audio,), daemon=True).start()

    def _handle_utterance(self, audio: np.ndarray) -> None:
        if self.stt is None:
            self.emit("error", {"message": "No speech-to-text provider is configured"})
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
        self.emit("transcript", {"role": "user", "text": text})
        self._run_turn(text)

    def _run_turn(self, user_text: str) -> None:
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
                for chunk in self.llm.stream(build_system_prompt(self.settings), self.conversation.messages):
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
                            pcm = self.tts.synthesize(s.text)
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
        try:
            # Wait for the first sentence before opening the HTTP stream so a cancelled or
            # empty reply never opens a playback.
            first = pcm_q.get()
            if first is not None and not turn.cancel.is_set():
                result = self.speak.stream_pcm(body(first), self.tts.samplerate)
                outcome = result.get("outcome", result.get("error", "unknown"))
                if not result.get("ok"):
                    self.emit("error", {"message": f"Speaker: {result.get('error')}"})
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
            if self._turn is turn:
                time.sleep(UNMUTE_DELAY)
                if self.mic:
                    self.mic.set_muted(False)
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
