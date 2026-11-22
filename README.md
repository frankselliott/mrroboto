# Ruko Companion

Turns a **Ruko 1088** robot into a conversational AI companion from a Mac: it listens through
the MacBook microphone, thinks with a language model, speaks through the robot's Bluetooth
speaker, and moves its arms and LED face to match what it is saying.

It is built on top of [Carle](https://github.com/benwilson/carle), Ben Wilson's reverse-engineered
Bluetooth protocol reference and control daemon for the 1088. Carle is used **unmodified** (a
plain clone in `carle/`, git-ignored) so upstream updates can be pulled at any time; everything
of ours is in `src/companion/`.

- [docs/architecture.md](docs/architecture.md) — how the pieces fit, and why these providers
- [docs/phase-1-hardware.md](docs/phase-1-hardware.md) — what was verified on the real robot

## Requirements

- macOS 12+ on Apple Silicon, Xcode Command Line Tools (`git`, `python3`)
- A Ruko 1088 switched on nearby, with its speaker paired in System Settings → Bluetooth
  (it appears as `JT_Speaker`)
- [`uv`](https://docs.astral.sh/uv/) — if you have no Homebrew: `python3 -m pip install --user uv`
  (lands in `~/Library/Python/3.9/bin`)

## Setup (once)

```bash
git clone https://github.com/frankselliott/mrroboto ruko-companion
cd ruko-companion
git clone https://github.com/benwilson/carle              # the protocol layer, kept pristine
(cd carle && uv sync --all-extras)                         # BLE + speak server + MCP
uv sync --all-extras                                       # the companion (Python 3.12)
scripts/fetch_models.sh                                    # local voice (Kokoro, ~350 MB)
clang -O2 -o macos/RukoCompanion.app/Contents/MacOS/RukoCompanion macos/launcher.c
codesign --force --deep --sign - macos/RukoCompanion.app   # the launcher that owns permissions
```

Then **double-click `macos/RukoCompanion.app`** (or `open macos/RukoCompanion.app`). The first
time, macOS asks to allow Bluetooth and the microphone for "RukoCompanion" — allow both. In the
window, open *Settings*, press *Scan*, and the robot's address is saved. Press *Start*.

### Why an app bundle?

macOS grants Bluetooth and microphone access to the *application* responsible for a process. A
bare terminal process is killed outright when it touches CoreBluetooth unless its parent app
declares Bluetooth usage. `macos/RukoCompanion.app` is a tiny ad-hoc-signed bundle whose
`Info.plist` declares both. Its executable is a small native binary (`macos/launcher.c`, built
with `clang`) that runs `launch.sh`, which starts the companion (or, with arguments, any command
under those permissions). The binary matters: a bundle whose main executable is a shell script
turns into `/bin/bash` in macOS's eyes, and the microphone prompt then never appears.

```bash
open -n -a macos/RukoCompanion.app --args carle scan                     # run Carle verbs
open -n -a macos/RukoCompanion.app --args uv run companion say "Hello"   # headless tests
```

Logs for everything launched this way are in `~/Library/Logs/RukoCompanion/`.

## Using it

| Control | What it does |
|---|---|
| **Start / Stop** | starts (or stops) the Carle daemon, speak server, microphone and AI providers |
| **Hold to talk** (or hold the space bar) | push-to-talk; release to send. Pressing while the robot speaks interrupts it |
| **Hands-free** | the robot listens continuously and replies when you stop talking |
| **Type a message** | skips the microphone — handy for testing |
| **STOP** | emergency stop: halts audio, clears the movement queue and brings the arms home |
| **Settings** | robot address, microphone, language model + API keys, voice, personality |
| **New conversation** | forgets the running conversation |

The transcript shows what the robot heard and what it said, live.

## Providers

Everything is pluggable from *Settings*; the defaults are free and local except the model.

| Role | Default | Alternatives |
|---|---|---|
| Language model | Claude `claude-sonnet-5-5` (needs `ANTHROPIC_API_KEY`) | Claude Haiku 5.5, OpenAI, any OpenAI-compatible local server (Ollama, LM Studio), *offline demo* (canned replies, no key) |
| Speech-to-text | `faster-whisper small.en`, local | OpenAI `whisper-1` |
| Text-to-speech | Kokoro, local, 24 kHz | OpenAI `gpt-4o-mini-tts`, macOS `say` |

API keys are stored in `~/Library/Application Support/RukoCompanion/secrets.env` (owner-only) and
never in this repository. Non-secret settings are in `config.yaml` next to it.

## How a reply becomes motion

The model is told it may place tags such as `[wave]`, `[think]`, `[celebrate]`, `[shrug]` before
a sentence. `companion/dialogue.py` strips them from the spoken text and `companion/behaviour.py`
maps each to a fixed, servo-safe step list from Carle's verified pose vocabulary. The model never
emits motor commands, nothing moves without a tag or a state change, and Carle's daemon still
enforces its one-joint-per-frame, 0.5 s-hold guard underneath.

LED faces follow the conversation state: smile while listening, concentrating while thinking,
talking mouth while speaking (switched to happy / laughing / sad for a tagged sentence).

## Wake word, owner and music

- **Hands-free with a wake word.** Tick *Hands-free* and the robot listens continuously but only
  answers when an utterance starts with the wake word (default "Mr Roboto"; Whisper's spellings
  such as "Mister Roboto" or "Mr Robot" all count). Say just the name and it answers "Yes?" and
  takes your next sentence as the question. Push-to-talk never needs the wake word.
- **Whose robot it is.** *Settings → Personality* has the robot's name, the owner's name, a free
  text "About the owner" box that goes straight into the robot's knowledge, and a pronunciation
  list (`Augie=Auggie` makes the voice say "Or-gee" rather than "Aj-ee").
- **Dancing.** Point *Settings → Music* at a folder of songs (any format macOS can play; the
  files' names become titles). Ask "play Intergalactic" or pick a song and press *Dance*: the
  song streams to the robot's speaker while it cycles through Carle's verified dance routines
  and arm moves, happy face on. Talking to it, pressing talk, or STOP ends the dance.

## Headless commands

```bash
uv run companion devices               # audio input devices
uv run companion scan                  # find the robot (run under the bundle, see above)
uv run companion say "text"            # speak through the robot with the configured voice
uv run companion turn "question" --then "follow-up" --interrupt-after 2
uv run companion listen --seconds 4    # record from the mic and print the transcript
uv run companion headless              # hands-free, no window, Ctrl-C to quit
uv run pytest                          # unit tests
```

Anything that needs Bluetooth must be launched through `macos/RukoCompanion.app`.

## Troubleshooting

- **Robot pill stays "connecting"** — is the robot on? Only one host can hold the BLE link; quit
  the Carle phone app. Carle reconnects on its own if the robot is power-cycled.
- **No sound** — the robot's speaker must be paired as `JT_Speaker`. If macOS made it the system
  default output, that is fine (the speak server targets it by name), but you may want your
  normal output back.
- **Silent `say`** — macOS's default Siri voice renders silence from the command line; the `say`
  provider uses a named voice (Samantha) for this reason.
- **Permission was denied** — System Settings → Privacy & Security → Bluetooth / Microphone →
  enable RukoCompanion, then relaunch.
