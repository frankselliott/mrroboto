import numpy as np

from companion.choreography import MIN_HOLD, Planner, beat_hold, build_routines, estimate_bpm


def test_bpm_from_a_click_track():
    sr = 24000
    for bpm in (96, 120, 140):
        t = np.zeros(sr * 30, dtype=np.int16)
        step = int(sr * 60 / bpm)
        for i in range(0, len(t) - 400, step):
            t[i : i + 400] = (np.sin(np.arange(400) * 0.3) * 12000).astype(np.int16)
        est = estimate_bpm(t, sr)
        assert abs(est - bpm) < 3, (bpm, est)
    assert estimate_bpm(np.zeros(sr * 2, dtype=np.int16), sr) == 110.0


def test_beat_hold_respects_servo_floor():
    assert beat_hold(120) == MIN_HOLD
    assert beat_hold(100) == 0.6
    assert beat_hold(170) >= MIN_HOLD


def test_routines_are_servo_safe_and_floor_moves_opt_in():
    for floor in (False, True):
        for r in build_routines(0.5, floor):
            assert r.duration > 2
            for s in r.steps:
                keys = [k for k in s if k in ("pose", "waist", "gesture", "pause", "travel")]
                assert len(keys) == 1, s
                if "pose" in s:
                    assert 1 <= s["pose"] <= 12 and s["hold"] >= MIN_HOLD
                if "waist" in s:
                    assert s["waist"] in (1, 2) and s["hold"] >= MIN_HOLD
                if "gesture" in s:
                    assert s["gesture"] in (19, 25, 27, 31, 33, 35, 37)
                if "travel" in s:
                    assert floor and r.floor and s["travel"]["speed"] <= 50 and s["travel"]["hold"] <= 2.5
    assert not any(r.floor for r in build_routines(0.5, False))


def test_planner_never_repeats_and_ramps_energy():
    p = Planner(build_routines(0.5), seed=1)
    names = [p.next(i / 30).name for i in range(30)]
    assert all(a != b for a, b in zip(names, names[1:]))
    energies = {r.name: r.energy for r in build_routines(0.5)}
    early = np.mean([energies[n] for n in names[:8]])
    late = np.mean([energies[n] for n in names[-8:]])
    assert late > early
