"""Speech-to-text providers behind one tiny interface."""

from __future__ import annotations

import io
import logging
import threading
from typing import Protocol

import numpy as np

from companion.config import Settings

_log = logging.getLogger(__name__)


class STT(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, samplerate: int) -> str: ...

    def warm_up(self) -> None: ...


class WhisperLocal:
    """faster-whisper on the CPU — free, private, good enough for short questions."""

    name = "faster-whisper (local)"

    def __init__(self, model_name: str = "small.en"):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel  # noqa: PLC0415 - heavy, lazy

            _log.info("loading whisper model %s", self.model_name)
            try:
                # Prefer the cached copy so start-up works offline and skips the hub round-trip.
                self._model = WhisperModel(
                    self.model_name, device="cpu", compute_type="int8", local_files_only=True
                )
            except Exception:  # noqa: BLE001 - not cached yet: download it
                self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray, samplerate: int) -> str:
        self.warm_up()
        if samplerate != 16000:
            audio = _resample(audio, samplerate, 16000)
        segments, _info = self._model.transcribe(
            audio.astype(np.float32),
            language="en",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


class OpenAIWhisper:
    name = "OpenAI whisper-1"

    def __init__(self, api_key: str, model: str = "whisper-1"):
        from openai import OpenAI  # noqa: PLC0415

        self._client = OpenAI(api_key=api_key)
        self.model = model

    def warm_up(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, samplerate: int) -> str:
        import soundfile as sf  # noqa: PLC0415

        buf = io.BytesIO()
        sf.write(buf, audio, samplerate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        buf.name = "speech.wav"
        result = self._client.audio.transcriptions.create(model=self.model, file=buf, language="en")
        return (result.text or "").strip()


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(audio) == 0:
        return audio
    n = int(len(audio) * dst / src)
    x_old = np.linspace(0, 1, len(audio), endpoint=False)
    x_new = np.linspace(0, 1, n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def make_stt(settings: Settings, secrets: dict[str, str]) -> STT:
    if settings.stt_provider == "openai":
        key = secrets.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OpenAI speech-to-text selected but OPENAI_API_KEY is not set")
        return OpenAIWhisper(key)
    return WhisperLocal(settings.stt_model or "small.en")
