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


def test_theme_selection_and_themed_routines_are_servo_safe():
    from companion.choreography import THEMES, energy_bands, routines_for, theme_for

    assert theme_for("Beastie Boys Intergalactic").name == "robot"
    assert theme_for("Tyrannosaurus Rex Dinosaur Song").name == "dinosaur"
    assert theme_for("Tung Tung Tung Sahur").name == "silly"
    assert theme_for("Pharrell Williams - Happy").name == "party"
    for theme in THEMES.values():
        for floor in (False, True):
            rs = routines_for(theme, 0.5, floor)
            assert len(rs) >= 8
            for r in rs:
                for s in r.steps:
                    if "pose" in s or "waist" in s:
                        assert s["hold"] >= MIN_HOLD
                    if "travel" in s:
                        assert floor and r.floor
        for band in theme.faces:
            assert band and all(f in (39, 41, 43, 45, 47) for f in band)
    sr = 24000
    quiet = np.zeros(sr * 4, dtype=np.int16)
    loud = (np.random.default_rng(0).standard_normal(sr * 4) * 8000).astype(np.int16)
    bands = energy_bands(np.concatenate([quiet, loud, quiet]), sr, 2.0)
    assert list(bands) == [0, 0, 2, 2, 0, 0]


def test_every_song_script_is_complete_and_safe():
    from companion.choreography import SCRIPTS, script_for, song_routines

    reg = song_routines(0.5)
    for script in SCRIPTS:
        assert script.sections[0].start == 0 and script.sections[-1].end >= 300
        for a, b in zip(script.sections, script.sections[1:]):
            assert a.end == b.start, (script.title, a, b)
        for sec in script.sections:
            assert sec.pool, sec
            for name in sec.pool + ((sec.signature,) if sec.signature else ()):
                assert name in reg, (script.title, name)
            assert all(f in (39, 41, 43, 45, 47) for f in sec.faces)
    for r in reg.values():
        for s in r.steps:
            if "pose" in s or "waist" in s or "travel" in s:
                assert s.get("hold", 0.5) >= MIN_HOLD, (r.name, s)
    titles = {
        "Styx Mr Roboto": "Styx — Mr Roboto", "Beastie Boys Intergalactic": "Beastie Boys — Intergalactic",
        "The Flaming Lips Yoshimi Battles the Pink Robots Pt 1": "The Flaming Lips — Yoshimi Battles the Pink Robots",
        "Austin Colin Big Bad Frog": "Austin & Colin — Big Bad Frog", "The Robot Song": "The Robot Song",
        "Tyrannosaurus Rex Dinosaur Song Pinkfong Songs for Children": "Pinkfong — Tyrannosaurus Rex",
        "Tung Tung Tung Sahur Ratatung": "Tung Tung Tung Sahur",
        "Robot Happy Birthday Song (Confetti)": "Robot Happy Birthday Song",
        "Birthday Robot Dance Song (Quantum Robot)": "Birthday Robot Dance Song",
        "Happy Birthday Augie": "Happy Birthday", "Pinkfong Baby Shark": "Pinkfong — Baby Shark",
    }
    for t, expected in titles.items():
        assert script_for(t) is not None and script_for(t).title == expected, t
    assert script_for("Pharrell Williams - Happy") is None
