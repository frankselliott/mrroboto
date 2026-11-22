"""Dance choreography: beat detection plus a library of multi-step routines on the beat.

Everything here is composed from Carle's hardware-verified primitives and obeys its servo
rules: one joint target changes per step, every hold is at least 0.5 s, canned ``0xB2``
routines are fired once, never streamed. The daemon's own guard enforces the same floor
underneath, so a mistake here would be slowed down rather than squeal the servos.

Joint cheat-sheet (``pose:N`` holds): 1/2 left arm forward up/down, 3/4 right arm forward,
5/6 left shoulder out/in, 7/8 right shoulder out/in, 9/10 left elbow up/down, 11/12 right
elbow up/down. ``waist`` 1 leans left (spring-return), 2 upright. Canned: 25, 31-38 torso
sways, 27 lean/bow, 19 arms home. Floor only: ``travel`` pivots (direction 3 vs 7 at low speed).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

MIN_HOLD = 0.5


# --- beat detection -------------------------------------------------------------------------


def estimate_bpm(pcm: np.ndarray, sr: int, lo: float = 70.0, hi: float = 170.0) -> float:
    """Rough tempo from an onset-energy autocorrelation. Falls back to 110 if unsure."""
    if len(pcm) < sr * 8:
        return 110.0
    x = pcm[: sr * 90].astype(np.float32)  # the first 90 s is plenty
    hop = sr // 100  # 10 ms frames
    n = len(x) // hop
    env = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
    onset = np.diff(env)
    onset[onset < 0] = 0
    onset -= onset.mean()
    if not np.any(onset):
        return 110.0
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1 :]
    min_lag, max_lag = int(100 * 60 / hi), int(100 * 60 / lo)
    seg = ac[min_lag:max_lag]
    if len(seg) == 0 or seg.max() <= 0:
        return 110.0
    lag = min_lag + int(np.argmax(seg))
    bpm = 6000.0 / lag
    # Prefer the danceable range: halve/double into [85, 150] when a multiple fits.
    while bpm > 150 and bpm / 2 >= lo:
        bpm /= 2
    while bpm < 85 and bpm * 2 <= hi:
        bpm *= 2
    return float(round(bpm, 1))


def beat_hold(bpm: float) -> float:
    """One move per beat if the beat is slow enough for the servos, else every second beat."""
    beat = 60.0 / max(bpm, 1.0)
    while beat < MIN_HOLD:
        beat *= 2
    return round(beat, 3)


# --- routines -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Routine:
    name: str
    steps: list[dict]
    energy: int  # 1 calm .. 3 wild
    floor: bool = False  # needs floor space (moves the robot)

    @property
    def duration(self) -> float:
        total = 0.0
        for s in self.steps:
            if "pose" in s or "waist" in s or "travel" in s:
                total += float(s.get("hold", MIN_HOLD))
            elif "pause" in s:
                total += float(s["pause"])
            elif "gesture" in s:
                total += 1.2
        return total


def _p(n: int, hold: float) -> dict:
    return {"pose": n, "hold": hold}


def _w(v: int, hold: float) -> dict:
    return {"waist": v, "hold": hold}


def _g(n: int) -> dict:
    return {"gesture": n}


def _pause(seconds: float) -> dict:
    return {"pause": seconds}


def _turn(direction: int, hold: float) -> dict:
    # Low speed = pivot roughly in place (docs/movement-vocabulary.md); short holds keep it small.
    return {"travel": {"direction": direction, "speed": 40, "mode": 1, "hold": hold}}


def build_routines(hold: float, floor_moves: bool = False) -> list[Routine]:
    """The routine library, with every hold scaled to the song's beat."""
    h = max(MIN_HOLD, hold)
    h2 = h * 2
    r: list[Routine] = [
        Routine("arms up pump", [
            _p(1, h), _p(3, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(9, h), _p(11, h),
            _p(10, h), _p(12, h), _p(2, h), _p(4, h),
        ], 3),
        Routine("windmill", [
            _p(5, h), _p(7, h), _p(1, h), _p(3, h), _pause(h), _p(6, h), _p(8, h), _p(2, h), _p(4, h),
        ], 2),
        Routine("disco point", [
            _p(3, h), _p(11, h), _p(12, h), _p(11, h), _p(12, h), _p(4, h),
            _p(1, h), _p(9, h), _p(10, h), _p(9, h), _p(10, h), _p(2, h),
        ], 3),
        Routine("shoulder shimmy", [
            _p(5, h), _p(6, h), _p(7, h), _p(8, h), _p(5, h), _p(6, h), _p(7, h), _p(8, h),
        ], 2),
        Routine("Y M C A", [
            _p(5, h), _p(1, h), _p(7, h), _p(3, h), _pause(h2), _p(6, h), _p(8, h), _p(2, h), _p(4, h),
        ], 2),
        Routine("clap along", [
            _p(1, h), _p(3, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(9, h), _p(11, h),
            _p(2, h), _p(4, h),
        ], 2),
        Routine("waist wiggle", [_w(1, h), _w(2, h), _w(1, h), _w(2, h), _w(1, h), _w(2, h)], 1),
        Routine("robot wave", [
            _p(1, h), _p(9, h), _p(7, h), _pause(h), _p(10, h), _p(9, h), _p(10, h), _p(8, h), _p(2, h),
        ], 2),
        Routine("torso groove", [_g(25), _pause(h2), _g(31), _pause(h2), _g(33)], 1),
        Routine("big sway", [_g(35), _pause(h2), _g(37), _pause(h2), _g(25)], 1),
        Routine("flex and bow", [_p(5, h), _p(9, h), _pause(h), _p(10, h), _p(6, h), _g(27)], 1),
        Routine("wave hello", [_p(3, h), _p(11, h), _p(12, h), _p(11, h), _p(12, h), _p(4, h)], 1),
        Routine("open arms", [_p(5, h), _p(7, h), _pause(h2), _w(1, h), _w(2, h), _p(6, h), _p(8, h)], 2),
        Routine("fist pumps", [
            _p(1, h), _p(9, h), _p(10, h), _p(9, h), _p(10, h), _p(2, h),
            _p(3, h), _p(11, h), _p(12, h), _p(11, h), _p(12, h), _p(4, h),
        ], 3),
    ]
    if floor_moves:
        r += [
            Routine("twist on the spot", [
                _p(5, h), _p(7, h), _turn(3, h), _turn(7, h), _turn(3, h), _turn(7, h), _p(6, h), _p(8, h),
            ], 3, floor=True),
            Routine("spin around", [_turn(3, h2 * 2), _pause(h), _g(25)], 3, floor=True),
        ]
    return r


class Planner:
    """Pick the next routine: no immediate repeats, energy rises through the song."""

    def __init__(self, routines: list[Routine], seed: int | None = None):
        self.routines = routines
        self._rng = random.Random(seed)
        self._last: str | None = None
        self.count = 0

    def next(self, progress: float) -> Routine:
        """``progress`` is 0..1 through the song; favour calm early and wild later."""
        target = 1 if progress < 0.25 else (2 if progress < 0.6 else 3)
        pool = [r for r in self.routines if r.name != self._last]
        weights = [3.0 if r.energy == target else (1.5 if abs(r.energy - target) == 1 else 0.5) for r in pool]
        choice = self._rng.choices(pool, weights=weights, k=1)[0]
        self._last = choice.name
        self.count += 1
        return choice


# --- themes: a style of dancing and a face palette per kind of song ---------------------------

FACE_SAD, FACE_LAUGH, FACE_THINK, FACE_TALK, FACE_HAPPY = 39, 41, 43, 45, 47


@dataclass(frozen=True)
class Theme:
    name: str
    description: str
    #: faces for quiet / medium / loud passages of the song (cycled within a band)
    faces: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    #: how long a face is held before the next in its band, in seconds
    face_period: float = 4.0


def robot_routines(h: float, floor: bool) -> list[Routine]:
    """Stop-motion: every move is a crisp two-beat hold, one joint at a time, like The Robot."""
    H = h * 2  # noqa: N806 - mechanical holds are double length
    r = [
        Routine("the robot", [
            _p(1, H), _p(9, H), _pause(h), _p(10, H), _p(2, H), _p(3, H), _p(11, H), _pause(h), _p(12, H), _p(4, H),
        ], 2),
        Routine("tin man", [_p(5, H), _pause(h), _p(9, H), _pause(h), _p(7, H), _pause(h), _p(11, H), _pause(h), _p(10, H), _p(12, H), _p(6, H), _p(8, H)], 2),
        Routine("robot arms", [_p(1, H), _p(3, H), _p(9, H), _p(11, H), _pause(H), _p(10, H), _pause(h), _p(12, H), _pause(h), _p(2, H), _p(4, H)], 3),
        Routine("servo tilt", [_w(1, H), _pause(h), _w(2, H), _pause(h), _w(1, H), _w(2, H)], 1),
        Routine("systems check", [_p(5, H), _pause(H), _p(6, H), _p(7, H), _pause(H), _p(8, H), _p(1, H), _pause(H), _p(2, H)], 1),
        Routine("power up", [_p(1, h), _p(3, h), _p(5, h), _p(7, h), _pause(H), _p(9, H), _p(11, H), _pause(H), _p(10, h), _p(12, h), _p(6, h), _p(8, h), _p(2, h), _p(4, h)], 3),
        Routine("robot wave", [_p(1, H), _p(9, H), _p(7, H), _pause(h), _p(10, H), _p(9, H), _p(10, H), _p(8, H), _p(2, H)], 2),
        Routine("glitch", [_p(9, h), _p(10, h), _p(9, h), _pause(H), _p(11, h), _p(12, h), _p(11, h), _pause(H), _p(10, h), _p(12, h)], 3),
        Routine("torso servo", [_g(25), _pause(H), _g(33)], 1),
    ]
    if floor:
        r.append(Routine("quarter turn", [_turn(3, H), _pause(H), _turn(7, H), _pause(h), _g(25)], 2, floor=True))
    return r


def dinosaur_routines(h: float, floor: bool) -> list[Routine]:
    """Stomps, tiny T-Rex arms, big roars and a swinging tail."""
    H = h * 2  # noqa: N806
    r = [
        Routine("tiny t-rex arms", [_p(9, h), _p(11, h), _p(1, h), _p(2, h), _p(3, h), _p(4, h), _p(1, h), _p(2, h), _p(3, h), _p(4, h), _p(10, h), _p(12, h)], 2),
        Routine("stomp stomp", [_w(1, H), _g(27), _pause(h), _w(1, H), _g(27), _pause(h), _w(2, h)], 2),
        Routine("big roar", [_p(5, h), _p(7, h), _p(9, h), _p(11, h), _pause(H * 2), _p(10, h), _p(12, h), _p(6, h), _p(8, h)], 3),
        Routine("tail swing", [_w(1, H), _w(2, H), _w(1, H), _w(2, H)], 1),
        Routine("chomp chomp", [_p(1, h), _p(3, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(2, h), _p(4, h)], 3),
        Routine("sniff the air", [_p(1, H), _p(9, H), _pause(H), _w(1, h), _w(2, h), _p(10, H), _p(2, H)], 1),
        Routine("thunder lizard", [_g(35), _pause(H), _g(27), _pause(H), _g(37)], 2),
        Routine("claws out", [_p(5, h), _p(9, h), _p(7, h), _p(11, h), _pause(H), _p(10, h), _p(12, h), _p(6, h), _p(8, h)], 2),
    ]
    if floor:
        r.append(Routine("lumber around", [_turn(3, H), _pause(h), _g(27), _turn(7, H)], 3, floor=True))
    return r


def silly_routines(h: float, floor: bool) -> list[Routine]:
    """Fast, floppy and ridiculous (still never faster than the servo floor)."""
    r = [
        Routine("flail", [_p(1, h), _p(5, h), _p(3, h), _p(7, h), _p(2, h), _p(6, h), _p(4, h), _p(8, h)], 3),
        Routine("jazz hands", [_p(1, h), _p(3, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(2, h), _p(4, h)], 3),
        Routine("big shrug", [_p(5, h), _p(7, h), _p(9, h), _p(11, h), _w(1, h), _w(2, h), _w(1, h), _w(2, h), _p(10, h), _p(12, h), _p(6, h), _p(8, h)], 2),
        Routine("wiggle wiggle", [_w(1, h), _w(2, h), _w(1, h), _w(2, h), _w(1, h), _w(2, h), _w(1, h), _w(2, h)], 2),
        Routine("shimmy shake", [_p(5, h), _p(6, h), _p(7, h), _p(8, h), _p(5, h), _p(6, h), _p(7, h), _p(8, h), _p(5, h), _p(7, h), _p(6, h), _p(8, h)], 3),
        Routine("silly sway", [_g(31), _pause(h * 2), _g(37), _pause(h * 2), _g(33), _pause(h * 2), _g(35)], 1),
        Routine("bow and pop", [_g(27), _pause(h), _p(1, h), _p(3, h), _p(2, h), _p(4, h), _g(27)], 2),
        Routine("one arm up", [_p(3, h), _p(11, h), _p(12, h), _p(11, h), _p(12, h), _p(11, h), _p(4, h), _p(1, h), _p(9, h), _p(10, h), _p(9, h), _p(2, h)], 2),
        Routine("sahur drum", [_p(9, h), _p(10, h), _p(11, h), _p(12, h), _p(9, h), _p(10, h), _p(11, h), _p(12, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h)], 3),
    ]
    if floor:
        r.append(Routine("dizzy spin", [_turn(3, h * 3), _pause(h), _turn(7, h * 3), _g(25)], 3, floor=True))
    return r


THEMES: dict[str, Theme] = {
    "robot": Theme(
        "robot", "stop-motion robot dance with a mechanical mouth",
        faces=((FACE_THINK,), (FACE_TALK, FACE_THINK), (FACE_TALK, FACE_HAPPY)), face_period=3.0,
    ),
    "dinosaur": Theme(
        "dinosaur", "stomping, roaring dinosaur",
        faces=((FACE_SAD,), (FACE_SAD, FACE_HAPPY), (FACE_LAUGH, FACE_HAPPY)), face_period=4.0,
    ),
    "silly": Theme(
        "silly", "super silly flailing",
        faces=((FACE_HAPPY,), (FACE_LAUGH, FACE_HAPPY), (FACE_LAUGH, FACE_THINK)), face_period=2.5,
    ),
    "party": Theme(
        "party", "general party dancing",
        faces=((FACE_HAPPY,), (FACE_HAPPY, FACE_TALK), (FACE_LAUGH, FACE_HAPPY)), face_period=4.0,
    ),
}

_THEME_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("robot", ("intergalactic", "robot", "roboto", "yoshimi", "machine", "android", "daft punk", "kraftwerk")),
    ("dinosaur", ("dinosaur", "t rex", "t-rex", "tyrannosaurus", "jurassic", "rex", "dino")),
    ("silly", ("tung", "sahur", "silly", "frog", "banana", "baby shark", "wiggle", "chicken")),
)


def theme_for(title: str) -> Theme:
    t = (title or "").lower()
    for name, words in _THEME_RULES:
        if any(w in t for w in words):
            return THEMES[name]
    return THEMES["party"]


def routines_for(theme: Theme, hold: float, floor: bool) -> list[Routine]:
    base = {
        "robot": robot_routines,
        "dinosaur": dinosaur_routines,
        "silly": silly_routines,
    }.get(theme.name)
    return base(hold, floor) if base else build_routines(hold, floor)


def energy_bands(pcm: np.ndarray, sr: int, window: float = 2.0) -> np.ndarray:
    """0 (quiet) / 1 (medium) / 2 (loud) per ``window`` seconds, from the song's loudness."""
    n = int(sr * window)
    if n <= 0 or len(pcm) < n:
        return np.array([1])
    frames = len(pcm) // n
    x = pcm[: frames * n].astype(np.float32).reshape(frames, n)
    rms = np.sqrt(np.mean(x * x, axis=1))
    lo, hi = np.percentile(rms, 33), np.percentile(rms, 66)
    bands = np.ones(frames, dtype=int)
    bands[rms <= lo] = 0
    bands[rms >= hi] = 2
    return bands


# --- bespoke scripts: one choreography per song, following its sections ----------------------


def song_routines(h: float) -> dict[str, Routine]:
    """Signature moves written for particular songs, plus every themed routine, by name."""
    H = h * 2  # noqa: N806
    extra = [
        # Styx — Mr Roboto
        Routine("domo bow", [_g(27), _pause(h), _p(1, h), _p(3, h), _pause(h), _p(2, h), _p(4, h), _g(27)], 2),
        Routine("secret secret", [_p(1, h), _p(9, h), _pause(H * 2), _p(10, h), _p(2, h), _w(1, h), _w(2, h)], 1),
        Routine("not a robot", [_p(5, h), _p(7, h), _pause(H), _p(1, h), _p(3, h), _pause(H), _p(2, h), _p(4, h), _p(6, h), _p(8, h)], 2),
        Routine("modern man", [_p(1, H), _p(3, H), _p(9, h), _p(11, h), _pause(H), _p(10, h), _p(12, h), _p(2, h), _p(4, h), _g(27)], 3),
        Routine("power down", [_p(1, H), _p(2, H), _w(1, H), _w(2, H), _pause(H), _g(19)], 1),
        # Beastie Boys — Intergalactic
        Routine("intergalactic punch", [_p(1, h), _p(2, h), _p(3, h), _p(4, h), _p(1, h), _p(9, h), _p(10, h), _p(2, h), _p(3, h), _p(11, h), _p(12, h), _p(4, h)], 3),
        Routine("planetary", [_p(5, h), _p(7, h), _pause(h), _w(1, h), _w(2, h), _w(1, h), _w(2, h), _p(6, h), _p(8, h)], 2),
        # The Flaming Lips — Yoshimi
        Routine("tai chi", [_p(5, H), _pause(h), _p(1, H), _pause(h), _p(6, H), _p(2, H), _p(7, H), _pause(h), _p(3, H), _p(8, H), _p(4, H)], 1),
        Routine("karate chop", [_p(3, h), _p(11, h), _p(12, h), _p(4, h), _p(1, h), _p(9, h), _p(10, h), _p(2, h), _p(3, h), _p(4, h)], 3),
        Routine("pink robot down", [_p(1, h), _p(3, h), _p(9, h), _p(11, h), _pause(h), _p(10, h), _p(12, h), _p(2, h), _p(4, h), _g(27)], 2),
        # Austin & Colin — Big Bad Frog
        Routine("frog crouch", [_p(9, H), _p(11, H), _pause(H * 2), _w(1, h), _w(2, h), _p(10, H), _p(12, H)], 1),
        Routine("frog hop", [_p(1, h), _p(3, h), _w(1, h), _p(2, h), _p(4, h), _w(2, h), _p(1, h), _p(3, h), _w(1, h), _p(2, h), _p(4, h), _w(2, h)], 3),
        Routine("tongue flick", [_p(3, h), _p(4, h), _p(3, h), _p(4, h), _p(11, h), _p(12, h)], 2),
        Routine("ribbit", [_p(9, h), _p(10, h), _p(9, h), _p(10, h), _p(11, h), _p(12, h), _p(11, h), _p(12, h)], 2),
        Routine("lily pad float", [_g(31), _pause(H), _w(1, H), _w(2, H), _g(33)], 1),
        # The Robot Song
        Routine("beep boop", [_p(1, h), _p(9, h), _p(10, h), _p(9, h), _p(10, h), _p(2, h)], 2),
        # Pinkfong — T-Rex
        Routine("dino wake", [_w(1, H), _w(2, H), _p(1, H), _p(9, H), _pause(H), _p(10, H), _p(2, H)], 1),
        # Birthday songs
        Routine("birthday clap", [_p(1, h), _p(3, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(9, h), _p(11, h), _p(2, h), _p(4, h)], 2),
        Routine("candles", [_p(1, H), _p(3, H), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(9, h), _p(11, h), _pause(H), _p(10, h), _p(12, h), _p(2, h), _p(4, h)], 2),
        Routine("make a wish", [_p(1, h), _p(9, h), _pause(H * 2), _p(10, h), _p(2, h), _p(1, h), _p(3, h), _pause(h), _p(2, h), _p(4, h)], 1),
        Routine("confetti", [_p(1, h), _p(3, h), _p(5, h), _p(7, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(6, h), _p(8, h), _p(2, h), _p(4, h), _g(25)], 3),
        # Baby Shark (ready for when the file arrives)
        Routine("baby chomp", [_p(9, h), _p(10, h), _p(9, h), _p(10, h), _p(9, h), _p(10, h), _p(11, h), _p(12, h), _p(11, h), _p(12, h)], 2),
        Routine("mommy chomp", [_p(1, h), _p(3, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(2, h), _p(4, h)], 2),
        Routine("daddy chomp", [_p(5, h), _p(7, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(9, h), _p(11, h), _p(10, h), _p(12, h), _p(6, h), _p(8, h)], 3),
        Routine("grandma shark", [_p(1, H), _p(9, H), _p(10, H), _p(9, H), _p(10, H), _p(2, H)], 1),
        Routine("lets go hunt", [_p(1, h), _p(3, h), _pause(h), _w(1, h), _w(2, h), _w(1, h), _w(2, h), _p(2, h), _p(4, h)], 2),
        Routine("run away", [_p(5, h), _p(6, h), _p(7, h), _p(8, h), _p(1, h), _p(2, h), _p(3, h), _p(4, h), _w(1, h), _w(2, h)], 3),
        Routine("safe at last", [_p(5, h), _p(7, h), _pause(H), _p(6, h), _p(8, h), _g(27)], 1),
        # Tung Tung Tung Sahur
        Routine("tung drum", [_p(9, h), _p(10, h), _p(9, h), _p(10, h), _p(9, h), _p(10, h), _pause(h), _p(11, h), _p(12, h), _p(11, h), _p(12, h), _p(11, h), _p(12, h)], 3),
    ]
    reg: dict[str, Routine] = {}
    for lib in (build_routines(h, True), robot_routines(h, True), dinosaur_routines(h, True), silly_routines(h, True), extra):
        for r in lib:
            reg.setdefault(r.name, r)
    return reg


@dataclass(frozen=True)
class Section:
    start: float
    end: float
    pool: tuple[str, ...]  # routine names to draw from
    faces: tuple[int, ...]  # cycled through this section
    signature: str | None = None  # fired when the section begins
    note: str = ""


@dataclass(frozen=True)
class SongScript:
    keywords: tuple[str, ...]
    title: str
    sections: tuple[Section, ...]
    face_period: float = 4.0

    def section_at(self, t: float) -> Section:
        for s in self.sections:
            if s.start <= t < s.end:
                return s
        return self.sections[-1]


T, K, P, L, H = FACE_TALK, FACE_THINK, FACE_HAPPY, FACE_LAUGH, FACE_SAD  # short names

SCRIPTS: tuple[SongScript, ...] = (
    SongScript(("confetti",), "Robot Happy Birthday Song", (
        Section(0, 12, ("systems check", "servo tilt", "make a wish"), (K, P), "power up", "the robots arrive"),
        Section(12, 28, ("birthday clap", "wave hello", "candles"), (P, T), "birthday clap", "happy birthday to you"),
        Section(28, 44, ("arms up pump", "confetti", "birthday clap"), (L, P), "confetti", "confetti!"),
        Section(44, 400, ("make a wish", "open arms", "flex and bow"), (P, L), "make a wish", "make a wish"),
    ), 3.0),
    SongScript(("quantum",), "Birthday Robot Dance Song", (
        Section(0, 20, ("systems check", "servo tilt", "the robot"), (K, T), "power up", "a birthday robot switches on"),
        Section(20, 52, ("robot arms", "beep boop", "birthday clap", "tin man"), (T, P), "beep boop", "robot verse"),
        Section(52, 70, ("arms up pump", "confetti", "candles"), (L, P), "confetti", "party chorus"),
        Section(70, 84, ("the robot", "tin man", "make a wish"), (T, K), None, "watch out!"),
        Section(84, 120, ("arms up pump", "open arms", "jazz hands", "birthday clap"), (L, P), "confetti", "big party"),
        Section(120, 400, ("make a wish", "flex and bow", "power down"), (P, K), "make a wish", "blow out the candles"),
    ), 3.0),
    SongScript(("birthday",), "Happy Birthday", (
        Section(0, 15, ("wave hello", "candles"), (P, T), "birthday clap", "happy birthday"),
        Section(15, 45, ("birthday clap", "arms up pump", "confetti", "open arms"), (L, P), "confetti", "celebrate"),
        Section(45, 400, ("make a wish", "flex and bow", "birthday clap"), (P, L), "make a wish", "make a wish"),
    ), 3.0),
    SongScript(("baby shark", "shark"), "Pinkfong — Baby Shark", (
        Section(0, 16, ("servo tilt", "tai chi"), (P,), None, "intro"),
        Section(16, 30, ("baby chomp",), (P, L), "baby chomp", "baby shark"),
        Section(30, 44, ("mommy chomp",), (P,), "mommy chomp", "mommy shark"),
        Section(44, 58, ("daddy chomp",), (L, P), "daddy chomp", "daddy shark"),
        Section(58, 72, ("grandma shark",), (P, K), "grandma shark", "grandma shark"),
        Section(72, 86, ("grandma shark", "tai chi"), (K, P), "grandma shark", "grandpa shark"),
        Section(86, 100, ("lets go hunt",), (K, H), "lets go hunt", "let's go hunt"),
        Section(100, 114, ("run away", "flail"), (L,), "run away", "run away!"),
        Section(114, 128, ("safe at last", "open arms"), (P,), "safe at last", "safe at last"),
        Section(128, 400, ("flex and bow", "wave hello"), (P, L), "flex and bow", "it's the end"),
    ), 3.0),
    SongScript(("mr roboto", "styx"), "Styx — Mr Roboto", (
        Section(0, 44, ("the robot", "servo tilt", "systems check"), (K, T), "domo bow", "robotic intro chant: slow, mechanical, bowing"),
        Section(44, 90, ("robot arms", "tin man", "the robot"), (T, K), "power up", "first verse"),
        Section(90, 130, ("secret secret", "glitch", "servo tilt"), (K,), "secret secret", "secret, secret, I've got a secret"),
        Section(130, 200, ("domo bow", "robot wave", "power up", "robot arms"), (T, P), "domo bow", "chorus: domo arigato"),
        Section(200, 260, ("not a robot", "open arms", "robot wave"), (P, T), "not a robot", "I'm not a robot without emotions"),
        Section(260, 290, ("servo tilt", "systems check", "tai chi"), (K,), None, "quiet breakdown"),
        Section(290, 400, ("modern man", "arms up pump", "domo bow"), (P, L), "modern man", "I am the modern man: big finish"),
    ), 3.0),
    SongScript(("intergalactic", "beastie"), "Beastie Boys — Intergalactic", (
        Section(0, 44, ("systems check", "servo tilt", "the robot"), (K, T), None, "vocoded intro: slow robot"),
        Section(44, 60, ("intergalactic punch", "robot arms"), (T, P), "intergalactic punch", "beat drops"),
        Section(60, 76, ("the robot", "tin man", "glitch"), (T,), None, "verse"),
        Section(76, 105, ("intergalactic punch", "planetary", "robot arms"), (P, T), "intergalactic punch", "chorus"),
        Section(105, 144, ("glitch", "robot wave", "tin man", "servo tilt"), (T, K), None, "verse"),
        Section(144, 170, ("intergalactic punch", "planetary", "power up"), (P, L), "intergalactic punch", "chorus"),
        Section(170, 192, ("the robot", "glitch", "robot arms"), (T,), None, "verse"),
        Section(192, 232, ("intergalactic punch", "planetary", "arms up pump"), (L, P), "intergalactic punch", "final chorus"),
        Section(232, 400, ("servo tilt", "systems check", "power down"), (K, T), "power down", "outro"),
    ), 3.0),
    SongScript(("yoshimi", "flaming lips"), "The Flaming Lips — Yoshimi Battles the Pink Robots", (
        Section(0, 48, ("tai chi", "torso groove", "servo tilt"), (K, P), None, "gentle verse: Yoshimi trains"),
        Section(48, 90, ("karate chop", "pink robot down", "tai chi"), (P, L), "karate chop", "chorus: fighting the pink robots"),
        Section(90, 116, ("tai chi", "torso groove", "wave hello"), (P, K), None, "verse"),
        Section(116, 185, ("karate chop", "pink robot down", "open arms"), (L, P), "karate chop", "big chorus"),
        Section(185, 400, ("tai chi", "big sway", "flex and bow"), (P, K), "flex and bow", "outro"),
    ), 4.0),
    SongScript(("frog",), "Austin & Colin — Big Bad Frog", (
        Section(0, 20, ("frog crouch", "ribbit"), (K, P), "frog crouch", "a frog on its lily pad"),
        Section(20, 76, ("frog hop", "tongue flick", "ribbit", "flail"), (L, P), "frog hop", "hopping about"),
        Section(76, 110, ("lily pad float", "frog crouch", "wiggle wiggle"), (P, K), "lily pad float", "quiet bit"),
        Section(110, 400, ("frog hop", "big shrug", "tongue flick", "bow and pop"), (L, P), "frog hop", "big bad finale"),
    ), 3.0),
    SongScript(("robot song",), "The Robot Song", (
        Section(0, 32, ("systems check", "servo tilt"), (K, T), "power up", "switching on"),
        Section(32, 90, ("beep boop", "robot arms", "the robot", "tin man"), (T, P), "beep boop", "I am a robot"),
        Section(90, 110, ("tin man", "servo tilt", "glitch"), (T, K), None, "verse"),
        Section(110, 160, ("power up", "beep boop", "robot wave", "the robot"), (P, T), "power up", "chorus"),
        Section(160, 192, ("arms up pump", "robot arms", "glitch"), (L, P), "arms up pump", "big section"),
        Section(192, 400, ("servo tilt", "systems check", "power down"), (K, T), "power down", "powering down"),
    ), 3.0),
    SongScript(("dinosaur", "tyrannosaurus", "t rex", "t-rex"), "Pinkfong — Tyrannosaurus Rex", (
        Section(0, 12, ("dino wake", "sniff the air"), (H, K), "dino wake", "the dinosaur wakes up"),
        Section(12, 40, ("stomp stomp", "big roar", "tiny t-rex arms"), (H, L), "big roar", "king of the dinosaurs"),
        Section(40, 64, ("tiny t-rex arms", "chomp chomp", "claws out", "tail swing"), (P, H), "chomp chomp", "verse"),
        Section(64, 90, ("big roar", "stomp stomp", "thunder lizard"), (L, H), "big roar", "roar!"),
        Section(90, 400, ("tail swing", "sniff the air", "flex and bow"), (P, K), "flex and bow", "settling down"),
    ), 3.0),
    SongScript(("tung", "sahur"), "Tung Tung Tung Sahur", (
        Section(0, 10, ("silly sway", "wiggle wiggle"), (P, K), None, "intro"),
        Section(10, 26, ("tung drum", "flail", "jazz hands"), (L, P), "tung drum", "tung tung tung"),
        Section(26, 46, ("wiggle wiggle", "big shrug", "silly sway"), (L, K), "big shrug", "silly bit"),
        Section(46, 72, ("tung drum", "shimmy shake", "flail"), (L, P), "tung drum", "tung tung tung"),
        Section(72, 84, ("silly sway", "one arm up"), (P, K), None, "breather"),
        Section(84, 400, ("tung drum", "sahur drum", "flail", "bow and pop"), (L, P), "tung drum", "finale"),
    ), 2.5),
)


def script_for(title: str) -> SongScript | None:
    t = (title or "").lower()
    for script in SCRIPTS:
        if any(k in t for k in script.keywords):
            return script
    return None
