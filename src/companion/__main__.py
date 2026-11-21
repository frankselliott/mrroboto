"""Entry point: `companion` opens the window; subcommands exist for headless testing."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from logging.handlers import RotatingFileHandler

from companion import config


def _setup_logging(verbose: bool) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh = RotatingFileHandler(config.LOG_DIR / "companion.log", maxBytes=2_000_000, backupCount=3)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if sys.stderr.isatty():
        # Under the launcher bundle stderr already lands in a log file; don't double-write.
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    for noisy in ("httpx", "httpcore", "urllib3", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _print_event(event: str, data: dict) -> None:
    if event in ("level",):
        return
    print(f"[{event}] {json.dumps(data)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="companion", description="Ruko Companion")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("ui", help="open the window (default)")
    sub.add_parser("devices", help="list audio input devices")
    sub.add_parser("scan", help="scan for the robot and save its address")
    say = sub.add_parser("say", help="speak text through the robot with the configured voice")
    say.add_argument("text", nargs="+")
    turn = sub.add_parser("turn", help="run one full conversational turn from typed text")
    turn.add_argument("text", nargs="+")
    turn.add_argument("--hold", type=float, default=0, help="keep running N seconds afterwards")
    turn.add_argument("--then", action="append", default=[], help="a follow-up message (repeatable)")
    turn.add_argument("--interrupt-after", type=float, default=0, help="interrupt the first reply after N seconds")
    listen = sub.add_parser("listen", help="record N seconds from the mic and transcribe")
    listen.add_argument("--seconds", type=float, default=4.0)
    sub.add_parser("headless", help="run the companion with no window (hands-free), until Ctrl-C")
    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    settings = config.load_settings()
    secrets = config.load_secrets()

    if args.cmd == "devices":
        from companion.audio import input_device_names

        for n in input_device_names():
            print(("* " if n == settings.mic_device else "  ") + n)
        return 0

    if args.cmd == "scan":
        from companion.app import scan_robots

        found = scan_robots()
        print(json.dumps(found, indent=2))
        if len(found) == 1:
            settings.robot_address = found[0]["address"]
            config.save_settings(settings)
            print(f"saved robot address {settings.robot_address}")
        return 0 if found else 1

    if args.cmd in ("say", "turn", "listen", "headless"):
        from companion.conversation import Companion

        c = Companion(settings, secrets, _print_event)
        _stop_on_signal(c)
        c.start()
        try:
            _wait_for(lambda: c.last_status.get("robot", {}).get("connected"), 20, "robot link")
            _wait_for(lambda: c.speak.alive(), 10, "speak server")
            if args.cmd == "say":
                from companion.dialogue import split_sentences

                c.tts.warm_up()
                text = " ".join(args.text)
                # Reuse the turn machinery with a model that just echoes the text.
                from companion.llm import Message

                class Echo:
                    name = "echo"

                    def stream(self, system, messages: list[Message]):
                        yield messages[-1].content

                c.llm = Echo()
                c.send_text(text)
                time.sleep(1.0)
                _wait_for(lambda: c.state.value in ("idle", "listening"), 120, "speech to finish")
            elif args.cmd == "turn":
                for i, text in enumerate([" ".join(args.text), *args.then]):
                    c.send_text(text)
                    if i == 0 and args.interrupt_after:
                        _wait_for(lambda: c.state.value == "speaking", 60, "speech to start")
                        time.sleep(args.interrupt_after)
                        print("[test] interrupting now", flush=True)
                        c.interrupt(reason="test")
                    time.sleep(1.0)
                    _wait_for(lambda: c.state.value in ("idle", "listening"), 180, "turn to finish")
                time.sleep(args.hold)
            elif args.cmd == "listen":
                print(f"recording {args.seconds}s — speak now")
                c.press_talk()
                time.sleep(args.seconds)
                audio = c.mic.stop_recording()
                t = time.monotonic()
                print("transcript:", repr(c.stt.transcribe(audio, 16000)), f"({time.monotonic() - t:.2f}s)")
            else:
                c.set_hands_free(True)
                print("hands-free; Ctrl-C to quit")
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            c.stop()
        return 0

    from companion.app import run_window

    run_window(settings, secrets)
    return 0


def _stop_on_signal(companion) -> None:
    """SIGTERM/SIGINT: stop the Carle children and bring the robot home before exiting."""
    import signal

    def handler(signum, frame):  # noqa: ARG001
        logging.getLogger(__name__).info("signal %s: shutting down", signum)
        companion.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _wait_for(cond, timeout: float, what: str) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            if cond():
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.25)
    print(f"warning: timed out waiting for {what}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
