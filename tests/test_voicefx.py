import numpy as np

from companion.voicefx import robotize


def test_robotize_shape_amount_and_level():
    sr = 24000
    t = np.arange(sr) / sr
    voice = (np.sin(2 * np.pi * 220 * t) * 12000).astype(np.int16)
    assert np.array_equal(robotize(voice, sr, 0.0), voice)
    out = robotize(voice, sr, 0.6)
    assert out.dtype == np.int16 and len(out) == len(voice)
    assert 0.85 * 32767 <= np.abs(out).max() <= 0.95 * 32767
    assert not np.array_equal(out, voice)
    assert len(robotize(np.zeros(0, dtype=np.int16), sr, 0.5)) == 0
