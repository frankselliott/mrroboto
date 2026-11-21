"""Text-to-speech providers. Each returns mono 16-bit PCM at its own sample rate."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from companion.config import MODELS_DIR, Settings

_log = logging.getLogger(__name__)

KOKORO_VOICES = (
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "am_echo", "am_eric",
    "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
)
OPENAI_VOICES = ("alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer")


class TTS(Protocol):
    name: str
    samplerate: int

    def synthesize(self, text: str) -> np.ndarray: ...

    def warm_up(self) -> None: ...


#: Target peak for synthesized speech (just under full scale) so quiet voices are not lost on
#: a small robot speaker.
TARGET_PEAK = 0.9


def _to_int16(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples[:, 0]
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak > 1e-4:
        samples = samples * (TARGET_PEAK / peak)
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)


class KokoroTTS:
    """Kokoro-82M via ONNX runtime — local, free, natural. 24 kHz output."""

    name = "Kokoro (local)"
    samplerate = 24000

    def __init__(self, voice: str = "af_heart", speed: float = 1.0):
        self.voice = voice if voice in KOKORO_VOICES else "af_heart"
        self.speed = max(0.6, min(1.5, speed))
        self._model = None
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from kokoro_onnx import Kokoro  # noqa: PLC0415 - heavy, lazy

            model = MODELS_DIR / "kokoro-v1.0.onnx"
            voices = MODELS_DIR / "voices-v1.0.bin"
            if not model.exists() or not voices.exists():
                raise RuntimeError(
                    "Kokoro model files missing from models/ — run scripts/fetch_models.sh"
                )
            _log.info("loading kokoro")
            self._model = Kokoro(str(model), str(voices))

    def synthesize(self, text: str) -> np.ndarray:
        self.warm_up()
        samples, sr = self._model.create(text, voice=self.voice, speed=self.speed, lang="en-us")
        if sr != self.samplerate:
            raise RuntimeError(f"unexpected kokoro sample rate {sr}")
        return _to_int16(samples)


class OpenAITTS:
    name = "OpenAI gpt-4o-mini-tts"
    samplerate = 24000

    def __init__(self, api_key: str, voice: str = "coral", speed: float = 1.0, model: str = "gpt-4o-mini-tts"):
        from openai import OpenAI  # noqa: PLC0415

        self._client = OpenAI(api_key=api_key)
        self.voice = voice if voice in OPENAI_VOICES else "coral"
        self.speed = max(0.5, min(2.0, speed))
        self.model = model

    def warm_up(self) -> None:
        pass

    def synthesize(self, text: str) -> np.ndarray:
        resp = self._client.audio.speech.create(
            model=self.model, voice=self.voice, input=text, response_format="pcm", speed=self.speed
        )
        return np.frombuffer(resp.content, dtype=np.int16)


class SayTTS:
    """macOS `say` with an explicit voice (the default Siri voice renders silence)."""

    name = "macOS say"
    samplerate = 22050

    def __init__(self, voice: str = "Samantha", speed: float = 1.0):
        self.voice = voice or "Samantha"
        self.rate = int(175 * max(0.5, min(2.0, speed)))

    def warm_up(self) -> None:
        pass

    def synthesize(self, text: str) -> np.ndarray:
        import soundfile as sf  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "say.wav"
            subprocess.run(
                [
                    "say", "-v", self.voice, "-r", str(self.rate), "-o", str(out),
                    "--file-format=WAVE", f"--data-format=LEI16@{self.samplerate}", text,
                ],
                check=True,
                capture_output=True,
            )
            data, _sr = sf.read(out, dtype="int16")
        return np.asarray(data, dtype=np.int16)


def make_tts(settings: Settings, secrets: dict[str, str]) -> TTS:
    if settings.tts_provider == "openai":
        key = secrets.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OpenAI speech selected but OPENAI_API_KEY is not set")
        return OpenAITTS(key, voice=settings.tts_voice, speed=settings.tts_speed)
    if settings.tts_provider == "say":
        voice = settings.tts_voice if settings.tts_voice and settings.tts_voice[0].isupper() else "Samantha"
        return SayTTS(voice, speed=settings.tts_speed)
    return KokoroTTS(voice=settings.tts_voice, speed=settings.tts_speed)
