"""The desktop window: a pywebview (WKWebView) page talking to Python directly.

No HTTP server is involved — the page calls ``window.pywebview.api.*`` and Python pushes
events with ``evaluate_js``. Nothing listens on any port.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
from pathlib import Path

from companion import config
from companion.audio import input_device_names
from companion.config import Settings
from companion.conversation import Companion
from companion.robot import CARLE_DIR, ROOT, find_uv
from companion.tts import KOKORO_VOICES, OPENAI_VOICES

_log = logging.getLogger(__name__)

UI_HTML = Path(__file__).parent / "ui" / "index.html"
_UUID_RE = re.compile(r"\b([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})\b")


def scan_robots(timeout: float = 12.0) -> list[dict]:
    """Run `carle scan` and return [{name, address}] for every JT_ peripheral seen."""
    cmd = [find_uv(), "run", "--project", str(CARLE_DIR), "carle", "scan", "--timeout", str(timeout)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout + 30)
    found: list[dict] = []
    name = None
    for line in out.stdout.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = _UUID_RE.search(line)
        if m and name:
            found.append({"name": name, "address": m.group(1)})
            name = None
        elif not line.startswith(" ") and line.startswith("JT_"):
            name = line.strip()
    if out.returncode != 0 and not found:
        _log.warning("scan failed (%s): %s", out.returncode, out.stderr.strip()[-400:])
    return found


class Api:
    """Methods callable from the page. Each returns JSON-serialisable data."""

    def __init__(self, companion: Companion):
        self.c = companion
        self._window = None

    def attach(self, window) -> None:
        self._window = window

    # --- state ----------------------------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "status": self.c.status(),
            "settings": self.c.settings.to_dict(),
            "secrets": config.secret_status(self.c.secrets),
            "transcript": [{"role": m.role, "content": m.content} for m in self.c.conversation.messages],
            "options": self.options(),
        }

    def options(self) -> dict:
        return {
            "songs": self.c.song_titles(),
            "personalities": list(config.PERSONALITIES) + ["custom"],
            "kokoro_voices": list(KOKORO_VOICES),
            "openai_voices": list(OPENAI_VOICES),
            "mics": input_device_names(),
            "llm_models": {
                "anthropic": ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"],
                "claude_code": ["sonnet", "haiku", "opus"],
                "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
                "openai_compatible": [],
                "scripted": [],
            },
        }

    # --- conversation control -------------------------------------------------------------

    def start(self) -> dict:
        self.c.start()
        return self.c.status()

    def stop(self) -> dict:
        self.c.stop()
        return self.c.status()

    def press_talk(self) -> None:
        self.c.press_talk()

    def release_talk(self) -> None:
        self.c.release_talk()

    def send_text(self, text: str) -> None:
        self.c.send_text(text)

    def set_hands_free(self, enabled: bool) -> None:
        self.c.set_hands_free(bool(enabled))
        config.save_settings(self.c.settings)

    def emergency_stop(self) -> None:
        self.c.emergency_stop()

    def interrupt(self) -> None:
        self.c.interrupt(reason="user")

    def reset_conversation(self) -> None:
        self.c.reset_conversation()

    def get_memories(self) -> list[str]:
        return list(self.c.memories)

    def save_memories(self, text: str) -> dict:
        self.c.save_memories((text or "").splitlines())
        return {"count": len(self.c.memories)}

    def play_spotify(self, query: str) -> dict:
        return {"ok": self.c.play_spotify(query)}

    def play_song(self, title: str) -> dict:
        ok = self.c.play_song(title)
        return {"ok": ok}

    # --- settings -------------------------------------------------------------------------

    def save_settings(self, data: dict) -> dict:
        new = Settings.from_dict({**self.c.settings.to_dict(), **(data or {})})
        config.save_settings(new)
        self.c.apply_settings(new, config.load_secrets())
        return {"settings": new.to_dict(), "status": self.c.status()}

    def set_secret(self, name: str, value: str) -> dict:
        value = (value or "").strip()
        config.set_secret(name, value)
        secrets = config.load_secrets()
        settings = self.c.settings
        # Saving a key while still on the offline demo almost certainly means "use this AI".
        if value and settings.llm_provider == "scripted":
            if name == "ANTHROPIC_API_KEY":
                settings = Settings.from_dict({**settings.to_dict(), "llm_provider": "anthropic", "llm_model": "claude-sonnet-5-5"})
            elif name == "OPENAI_API_KEY":
                settings = Settings.from_dict({**settings.to_dict(), "llm_provider": "openai", "llm_model": "gpt-4o-mini"})
            config.save_settings(settings)
        self.c.apply_settings(settings, secrets)
        return config.secret_status(secrets)

    def scan(self) -> dict:
        found = scan_robots()
        if len(found) == 1 and not self.c.settings.robot_address:
            self.save_settings({"robot_address": found[0]["address"]})
        return {"found": found}

    def open_logs(self) -> None:
        subprocess.Popen(["open", str(config.LOG_DIR)])


def push(window, event: str, data: dict) -> None:
    try:
        window.evaluate_js(f"window.onCompanionEvent({json.dumps(event)}, {json.dumps(data)})")
    except Exception:  # noqa: BLE001 - window may be closing
        pass


def run_window(settings: Settings, secrets: dict[str, str], autostart: bool = True) -> None:
    import webview  # noqa: PLC0415 - imports Cocoa; keep headless paths free of it

    window_ref: dict = {}
    pending: list[tuple[str, dict]] = []
    lock = threading.Lock()

    def emit(event: str, data: dict) -> None:
        w = window_ref.get("w")
        if w is None:
            with lock:
                pending.append((event, data))
            return
        push(w, event, data)

    companion = Companion(settings, secrets, emit)
    api = Api(companion)
    window = webview.create_window(
        "Ruko Companion",
        html=UI_HTML.read_text(),
        js_api=api,
        width=1040,
        height=760,
        min_size=(760, 560),
    )
    api.attach(window)

    def on_loaded():
        window_ref["w"] = window
        with lock:
            for ev in pending:
                push(window, *ev)
            pending.clear()
        if autostart and settings.robot_address:
            threading.Thread(target=companion.start, daemon=True).start()

    def on_closing():
        threading.Thread(target=companion.stop, daemon=True).start()

    window.events.loaded += on_loaded
    window.events.closing += on_closing
    webview.start(debug=False)
