# Ruko Companion — architecture

Goal: talking to the Ruko 1088 should feel like talking to a real AI robot. The MacBook listens,
thinks and renders speech; the robot speaks (Bluetooth A2DP, `JT_Speaker`) and moves (BLE, via
Carle). Carle is used unmodified as a dependency; everything here lives in `src/companion/`.

## Processes

```
RukoCompanion.app (launcher; owns Bluetooth + microphone permission)
 └─ companion (Python 3.12, pywebview window)
     ├─ carle daemon start <uuid> --foreground      BLE link holder, servo-safe queue, heartbeat
     ├─ carle speak-server --no-animate             audio → JT_Speaker, /speak/stream + /speak/stop
     └─ in-process: mic → STT → LLM → TTS → speak server, behaviour engine → daemon socket
```

The companion spawns and supervises both Carle processes (restart on crash, status polling),
so the user never opens a terminal. The daemon socket is relative (`.carle/daemon.sock`), so
the companion always runs with the project root as working directory.

## Conversation pipeline

1. **Listen** — `sounddevice` opens `MacBook Air Microphone` explicitly (the system default
   input is the robot's 8 kHz HFP mic). Push-to-talk records until release; hands-free mode
   adds an energy/VAD gate. Capture is **disabled while the robot speaks** (half-duplex), which
   is the echo protection: the mic is on the laptop and the speaker is on the robot, so simply
   not listening while talking is robust and free.
2. **Transcribe** — `STT` provider interface. Default: `faster-whisper small.en` on CPU
   (local, free, ~1 s for a short utterance). Alternative: OpenAI `whisper-1`.
3. **Think** — `LLM` provider interface with streaming. Default: Anthropic Claude
   (`claude-sonnet-5-5`; `claude-haiku-5-5` for lowest latency). Alternatives: OpenAI
   (`gpt-4o-mini`), any OpenAI-compatible endpoint (Ollama, LM Studio), and a `scripted`
   provider for offline testing of the whole pipeline. Conversation memory is the full message
   list for the session, trimmed to a budget, plus a personality system prompt.
4. **Express** — the model is told to place gesture/emotion tags such as `[wave]`, `[celebrate]`,
   `[think]`, `[happy]` inline. `dialogue.py` strips them from the spoken text and attaches each
   to the sentence it appears in. The model never emits motor commands; `behaviour.py` maps
   each tag to a fixed, servo-safe step list taken from Carle's documented vocabulary.
5. **Speak** — `TTS` provider interface producing 16-bit PCM. Default: Kokoro (local ONNX,
   24 kHz, free, natural). Alternatives: OpenAI `gpt-4o-mini-tts` (`pcm`), ElevenLabs
   (`pcm_24000`), macOS `say -v Samantha` (zero dependencies, fallback). Sentences are
   synthesized one at a time and written into **one** `POST /speak/stream?format=raw` request,
   paced to real time, so the robot starts talking after the first sentence rather than after
   the whole reply, and the behaviour engine fires a sentence's gestures as its audio is
   written (≈0.3 s ahead of the speaker).
6. **Interrupt** — pressing talk (or saying something in hands-free mode) while the robot speaks
   cancels generation, `POST /speak/stop`, and `stop` on the daemon (arms home, face cleared).

## Behaviour engine

States and faces (Carle expression codes, hardware-verified): idle → robot's own face;
listening → 47 (smile); thinking → 43 (concentrating); speaking → 45 (talking mouth), switched
to 47/41/39 for a sentence tagged happy/laugh/sad. Gestures are an **approved list** only:
wave (right arm raise + two forearm bobs), goodbye wave (left arm), celebrate (both arms up),
shrug, think (hand near face), explain (small forward arm motion), present, nod, sway (canned
code 25), bow (canned 27). Each is a `pose:`/`gesture:` list the daemon already knows how to
run within its one-joint-per-frame, 0.5 s-hold guard. Nothing moves unless a tag asks for it or a
state changes; there is no random idle animation.

## Interface

A `pywebview` native window (WKWebView) loading a single local HTML page; Python exposes an API
to the page (start/stop, talk, settings, emergency stop) and pushes events (state, transcript,
connection status, levels). No network server at all — nothing listens on any port except
Carle's loopback speak server on 127.0.0.1:8081.

## Provider recommendation

| Role | Recommended | Why | Cost |
|---|---|---|---|
| Conversation | **Claude `claude-sonnet-5-5`** (Haiku 5.5 when latency matters more) | best conversational quality and instruction-following for the inline-tag protocol; streaming; provider swap is one class | ~$0.01–0.03 per reply |
| Speech-to-text | **faster-whisper `small.en`, local** | free, private, no network, accurate for short questions; Whisper API as a drop-in | $0 |
| Text-to-speech | **Kokoro, local** | free, natural voice, ~real-time on Apple Silicon, raw PCM fits `/speak/stream`; OpenAI TTS as a higher-quality paid option | $0 |

Reasoning: a speech-to-speech API (OpenAI Realtime) gives the lowest latency but hides the text
we need for gesture tags and memory, locks the stack to one vendor, and costs several times
more. Splitting STT / LLM / TTS keeps each piece swappable from the settings panel and leaves
only the language model as a paid call.

## Security and privacy

- API keys live in `~/Library/Application Support/RukoCompanion/secrets.env` (mode 600) or the
  environment, never in source or the repo. The settings panel writes them there.
- Audio buffers are processed in memory and discarded; nothing is written to disk unless
  `debug.keep_audio` is set.
- No listening sockets are created by the companion; Carle's speak server is loopback-only.
- Movements are restricted to the approved list; the model has no path to raw frames.
