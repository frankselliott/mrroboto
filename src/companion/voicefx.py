"""A robot voice effect applied to synthesized speech (16-bit mono PCM).

Three ingredients, each scaled by ``amount`` (0 = untouched, 1 = full robot):
- ring modulation with a low sine (the classic metallic buzz),
- a short feedback echo (a few milliseconds) for a tinny, resonant timbre,
- gentle saturation so it sounds electronic rather than muffled.
Intelligibility matters more than the effect: the dry voice is always mixed back in.
"""

from __future__ import annotations

import numpy as np

RING_HZ = 42.0
COMB_MS = 2.3


def robotize(pcm: np.ndarray, sr: int, amount: float = 0.5) -> np.ndarray:
    amount = float(max(0.0, min(1.0, amount)))
    if amount <= 0.0 or len(pcm) == 0:
        return pcm
    x = pcm.astype(np.float32) / 32768.0
    n = len(x)
    t = np.arange(n, dtype=np.float32) / sr

    # Ring modulation: multiply by a low sine, mix with the dry signal.
    ring = x * np.sin(2.0 * np.pi * RING_HZ * t)
    wet = (1.0 - 0.6 * amount) * x + (0.9 * amount) * ring

    # Short feedback comb filter: y[n] = wet[n] + g * y[n - d]  (metallic resonance).
    d = max(1, int(sr * COMB_MS / 1000.0))
    g = 0.35 * amount
    y = wet.copy()
    if g > 0:
        # Vectorised block recursion: process in chunks of d samples.
        for start in range(d, n, d):
            end = min(n, start + d)
            y[start:end] += g * y[start - d : end - d]

    # Gentle saturation and normalisation.
    y = np.tanh(y * (1.0 + 1.5 * amount)) / np.tanh(1.0 + 1.5 * amount)
    peak = float(np.max(np.abs(y))) if n else 0.0
    if peak > 1e-4:
        y = y * (0.9 / peak)
    return (y * 32767.0).astype(np.int16)
