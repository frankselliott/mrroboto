"""Microphone capture: push-to-talk and a simple hands-free voice gate.

One ``sounddevice`` input stream stays open on the chosen microphone for the life of the app.
Its callback keeps a level meter current and, while *recording*, appends blocks to a buffer.
While the robot speaks, capture is **muted** (blocks are dropped) — that is the echo protection:
the mic is on the laptop, the speaker is on the robot, so simply not listening while talking is
enough and costs nothing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np
import sounddevice as sd

_log = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # what Whisper wants; the device is resampled by CoreAudio/PortAudio
BLOCK = 1600  # 100 ms


def find_input_device(name_substring: str) -> int | None:
    """Index of the first input device whose name contains ``name_substring``."""
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name_substring.lower() in d["name"].lower():
            return i
    return None


def input_device_names() -> list[str]:
    return [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]


class Microphone:
    def __init__(self, device_name: str, on_level: Callable[[float], None] | None = None):
        self.device_name = device_name
        self.device_index = find_input_device(device_name)
        self.on_level = on_level
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._recording = False
        self._muted = False
        self._chunks: list[np.ndarray] = []
        self.level = 0.0
        self.noise_floor = 0.004
        self._last_level_report = 0.0
        self._listener: Callable[[np.ndarray, float], None] | None = None

    # --- stream lifecycle -----------------------------------------------------------------

    def open(self) -> None:
        if self._stream is not None:
            return
        if self.device_index is None:
            raise RuntimeError(f"input device containing {self.device_name!r} not found")
        self._stream = sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        _log.info("microphone open: %s", sd.query_devices(self.device_index)["name"])

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # --- control --------------------------------------------------------------------------

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    @property
    def muted(self) -> bool:
        return self._muted

    def start_recording(self) -> None:
        with self._lock:
            self._chunks = []
            self._recording = True

    def stop_recording(self) -> np.ndarray:
        with self._lock:
            self._recording = False
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    def set_listener(self, fn: Callable[[np.ndarray, float], None] | None) -> None:
        """Receive every unmuted block (hands-free mode)."""
        self._listener = fn

    # --- callback -------------------------------------------------------------------------

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        mono = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(mono * mono))) if frames else 0.0
        self.level = rms
        # Track the quiet-room floor slowly so the hands-free gate adapts to the room.
        if rms < self.noise_floor * 3:
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * max(rms, 1e-4)
        now = time.monotonic()
        if self.on_level and now - self._last_level_report > 0.1:
            self._last_level_report = now
            try:
                self.on_level(rms)
            except Exception:  # noqa: BLE001 - UI reporting must never break capture
                pass
        if self._muted:
            return
        if self._recording:
            with self._lock:
                self._chunks.append(mono)
        if self._listener is not None:
            self._listener(mono, rms)


class VoiceGate:
    """Energy-based hands-free gate: calls ``on_utterance(audio)`` when speech ends.

    Speech starts after ``start_blocks`` consecutive blocks above threshold and ends after
    ``end_silence`` seconds below it. Utterances shorter than ``min_duration`` are dropped
    (coughs, chair squeaks).
    """

    def __init__(
        self,
        mic: Microphone,
        on_utterance: Callable[[np.ndarray], None],
        threshold: float = 0.012,
        end_silence: float = 0.8,
        min_duration: float = 0.5,
        max_duration: float = 20.0,
        start_blocks: int = 2,
    ):
        self.mic = mic
        self.on_utterance = on_utterance
        self.threshold = threshold
        self.end_silence = end_silence
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.start_blocks = start_blocks
        self._active = False
        self._above = 0
        self._silence = 0.0
        self._buf: list[np.ndarray] = []
        self._pre: list[np.ndarray] = []  # a little audio before the trigger
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True
        self._reset()
        self.mic.set_listener(self._on_block)

    def disable(self) -> None:
        self.enabled = False
        self.mic.set_listener(None)
        self._reset()

    def _reset(self) -> None:
        self._active = False
        self._above = 0
        self._silence = 0.0
        self._buf = []
        self._pre = []

    def _on_block(self, block: np.ndarray, rms: float) -> None:
        if not self.enabled:
            return
        block_s = len(block) / SAMPLE_RATE
        gate = max(self.threshold, self.mic.noise_floor * 4)
        if not self._active:
            self._pre.append(block)
            self._pre = self._pre[-3:]
            if rms > gate:
                self._above += 1
                if self._above >= self.start_blocks:
                    self._active = True
                    self._buf = list(self._pre)
                    self._silence = 0.0
            else:
                self._above = 0
            return
        self._buf.append(block)
        if rms > gate:
            self._silence = 0.0
        else:
            self._silence += block_s
        duration = sum(len(b) for b in self._buf) / SAMPLE_RATE
        if self._silence >= self.end_silence or duration >= self.max_duration:
            audio = np.concatenate(self._buf)
            self._reset()
            if duration - self._silence >= self.min_duration:
                self.on_utterance(audio)
