"""Settings and secrets.

Settings (non-secret) live in a YAML file under the user's Application Support folder; API keys
live in a separate ``secrets.env`` file with mode 600, or in the process environment. Neither
is ever written into the project tree, so nothing secret can end up in git.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

APP_DIR = Path.home() / "Library" / "Application Support" / "RukoCompanion"
CONFIG_PATH = APP_DIR / "config.yaml"
SECRETS_PATH = APP_DIR / "secrets.env"
LOG_DIR = Path.home() / "Library" / "Logs" / "RukoCompanion"

#: The project root: <root>/src/companion/config.py
ROOT = Path(__file__).resolve().parents[2]
CARLE_DIR = ROOT / "carle"
MODELS_DIR = ROOT / "models"

SECRET_NAMES = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY")

PERSONALITIES: dict[str, str] = {
    "friendly": (
        "You are a small friendly robot companion with LED eyes and two little arms. "
        "You are warm, curious, upbeat and genuinely interested in the person you are talking "
        "to. You speak naturally, like a thoughtful friend, and you enjoy a light joke."
    ),
    "witty": (
        "You are a small robot with a dry, quick wit. You are kind underneath the banter, "
        "never mean, and you love clever observations. Keep the jokes short."
    ),
    "calm": (
        "You are a gentle, calm robot companion. You speak slowly and clearly, you are "
        "patient and reassuring, and you never rush the person you are talking to."
    ),
    "professor": (
        "You are a small robot who adores explaining how things work. You are precise, "
        "enthusiastic about ideas, and you check the listener is following."
    ),
}


@dataclass
class Settings:
    # robot
    robot_address: str = ""
    speaker_device: str = "JT_Speaker"
    speak_port: int = 8081
    animate: bool = True
    # audio in
    mic_device: str = "MacBook Air Microphone"
    hands_free: bool = True
    vad_threshold: float = 0.012
    # speech to text
    stt_provider: str = "local"  # local | openai
    stt_model: str = "small.en"
    # language model
    llm_provider: str = "anthropic"  # anthropic | openai | openai_compatible | scripted
    llm_model: str = "claude-sonnet-5-5"
    llm_base_url: str = ""  # for openai_compatible (Ollama, LM Studio, ...)
    max_history_turns: int = 40
    max_reply_tokens: int = 400
    # text to speech
    tts_provider: str = "kokoro"  # kokoro | openai | say
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    # personality
    personality: str = "friendly"
    personality_custom: str = ""
    robot_name: str = "Mr Roboto"
    user_name: str = ""
    owner_name: str = "Augie"
    owner_profile: str = ""  # free text: everything the robot should know about its owner
    pronunciations: str = "Augie=Auggie"  # one "word=spelling-to-speak" per line
    # wake word (hands-free mode)
    wake_word: str = "Mr Roboto"
    wake_word_enabled: bool = True
    wake_window_seconds: float = 10.0
    follow_up_seconds: float = 15.0  # after a reply, keep listening this long without the wake word
    # conversation initiative (hands-free)
    auto_greet: bool = True  # greet when the robot comes online
    idle_nudge_minutes: float = 4.0  # say something after this much silence (0 = never)
    idle_nudge_max: int = 2  # at most this many nudges per silence
    # music
    music_dir: str = "~/Downloads/Robot songs"
    # privacy / debug
    keep_audio: bool = False

    def pronunciation_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in (self.pronunciations or "").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() and v.strip():
                    out[k.strip()] = v.strip()
        return out

    def personality_text(self) -> str:
        if self.personality == "custom" and self.personality_custom.strip():
            return self.personality_custom.strip()
        return PERSONALITIES.get(self.personality, PERSONALITIES["friendly"])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Settings:
        known = {f.name: f for f in fields(cls)}
        kwargs = {}
        for k, v in (data or {}).items():
            if k in known:
                kwargs[k] = _coerce(known[k].type, v)
        return cls(**kwargs)


def _coerce(type_name, value):
    # dataclass field types are strings under `from __future__ import annotations`
    t = type_name if isinstance(type_name, str) else getattr(type_name, "__name__", "")
    try:
        if t == "bool":
            return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
        if t == "int":
            return int(value)
        if t == "float":
            return float(value)
        if t == "str":
            return "" if value is None else str(value)
    except (TypeError, ValueError):
        pass
    return value


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    if path.exists():
        with path.open() as fh:
            return Settings.from_dict(yaml.safe_load(fh) or {})
    return Settings()


def save_settings(settings: Settings, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(settings.to_dict(), fh, sort_keys=False)


# --- secrets --------------------------------------------------------------------------------

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$")


def _parse_env(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key] = val
    return out


def load_secrets(path: Path = SECRETS_PATH) -> dict[str, str]:
    """Secrets from the env file, with process environment variables as a fallback."""
    secrets: dict[str, str] = {}
    if path.exists():
        secrets.update(_parse_env(path.read_text()))
    for name in SECRET_NAMES:
        if name not in secrets and os.environ.get(name):
            secrets[name] = os.environ[name]
    return secrets


def set_secret(name: str, value: str, path: Path = SECRETS_PATH) -> None:
    """Write one secret (or remove it when value is empty) with owner-only permissions."""
    if name not in SECRET_NAMES:
        raise ValueError(f"unknown secret {name!r}")
    current = _parse_env(path.read_text()) if path.exists() else {}
    if value:
        current[name] = value
    else:
        current.pop(name, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{k}={v}\n" for k, v in current.items())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(body)
    os.chmod(path, 0o600)


def secret_status(secrets: dict[str, str]) -> dict[str, bool]:
    """Which secrets are present — never the values — for the settings panel."""
    return {name: bool(secrets.get(name)) for name in SECRET_NAMES}
