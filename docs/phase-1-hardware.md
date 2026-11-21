# Phase 1 — hardware validation (2026-11-22)

Everything below was run against the real robot from this Mac using the unmodified Carle
checkout at `carle/` (commit `e7d8a2e`). Nothing in the robot's firmware was touched and every
movement used Carle's documented servo-safe steps.

## Environment

| Item | Finding |
|---|---|
| Mac | MacBook Air (arm64), macOS 26.5.2, Command Line Tools present, no Homebrew |
| Python | System 3.9 only. `uv` installed per-user via pip (`~/Library/Python/3.9/bin/uv`); Carle's venv is Python 3.15 |
| Carle | `uv sync --all-extras` succeeded (BLE, speak, MCP extras). `carle --help` lists all verbs |
| Bluetooth | Controller on. `JT_Speaker` was already paired and was the **system default output and input** |
| Audio devices | PortAudio sees `JT_Speaker` (out), `JT_Speaker` HFP mic (in, 8 kHz), `MacBook Air Microphone`, `MacBook Air Speakers` |
| AI keys | None present in the environment or shell rc files |

## Bluetooth permission on macOS

`carle scan` launched from Claude Code's shell was killed by macOS (exit 134, no prompt). The
responsible app for that shell is the `com.anthropic.claude-code` helper bundle, whose Info.plist
has a microphone usage description but **no `NSBluetoothAlwaysUsageDescription`**, so
CoreBluetooth aborts the process instead of asking.

Fix: `macos/RukoCompanion.app`, an ad-hoc-signed launcher bundle declaring Bluetooth and
microphone usage. Anything that touches the robot runs under it:

```bash
open -n -a macos/RukoCompanion.app --args carle scan            # logs → ~/Library/Logs/RukoCompanion/
open -n -a macos/RukoCompanion.app --args carle daemon start <uuid> --foreground
```

macOS prompted once for Bluetooth for "RukoCompanion" and it was granted. The same bundle will
own the microphone permission for the companion app, which is why it exists now rather than later.

## Results

| # | Test | Command | Result |
|---|---|---|---|
| 1 | Scan | `carle scan` | Found **`JT_19120E`**, UUID `831EA70E-A508-BD19-C5EC-BEB177B64200`, RSSI −56 |
| 2 | GATT | `carle info <uuid>` | Exactly one service `AE00` with `AE01` write-without-response and `AE02` notify — identical to Carle's protocol reference |
| 3 | Connect / hold link | `carle daemon start <uuid> --foreground` | `carle status` → `connected; battery unknown; doing nothing` within 10 s (battery unknown is expected: no battery characteristic exists) |
| 4 | Speaker pairing | — | Already paired as `JT_Speaker`; visible to PortAudio as output device 1 |
| 5 | Safe arm movement | `carle queue gesture:19` then `carle queue wave` | Both accepted (1 + 6 steps); queue drained in ~3 s; status clean afterwards. Visual confirmation by operator pending |
| 6 | Audio via daemon `say:` step | `carle queue say:...` | Spawned, but **silent**: macOS `say` with the default (Siri) voice renders 0.005 s of audio. `say -v Samantha` works. Not a Carle bug as such, but its `say:` step is unusable on this Mac without a voice override |
| 7 | Audio via speak server | `carle speak-server` + `POST /speak/clip` (3.5 s tone) | `{"ok": true, "outcome": "completed"}` in 4.2 s wall time — real-time playback through the targeted `JT_Speaker` device |
| 8 | Speech + face + arms together | `POST /speak/clip` with an 11.5 s Samantha WAV | Completed in 12.2 s; speak server held talking face 45 and pulsed gestures 1/3 via the daemon, then returned face/arms to neutral (`face:0`, `gesture:19`). Daemon state clean afterwards |
| 9 | Interrupt | 12.8 s clip, `POST /speak/stop` at 4 s | Stop returned `{"stopped": true}`, clip returned `outcome: "stopped"`, robot returned to neutral |

The operator (Elliott) should confirm by eye and ear: the wave in test 5, the melody in test 7,
the speech + animation in test 8, and the cut-off in test 9. Everything the software can
observe is consistent with the Carle hardware-validation records.

## What this means for the companion

- **Reuse as-is:** the daemon (link holding, heartbeat, servo-safe queue, reconnect/resume),
  the pose/gesture/face vocabulary, the speak server's audio path (`/speak/clip`,
  `/speak/stream` with raw PCM, `/speak/stop`, device targeting, 409/503 semantics).
- **Run the speak server with `--no-animate`** in the companion and drive the talking face and
  gestures from our own behaviour engine over the daemon socket, so expressions match what is
  being said rather than alternating gestures 1 and 3.
- **Microphone:** never use the system default input (it is the robot's 8 kHz HFP mic); open
  `MacBook Air Microphone` explicitly.
- **TTS:** do not rely on `say`; the companion renders speech itself and streams PCM to the
  speak server.
- **Both Carle processes must run under `RukoCompanion.app`** and with the project root as
  working directory, because the daemon socket path is relative (`.carle/daemon.sock`).
