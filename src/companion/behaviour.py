"""Behaviour engine — the approved vocabulary of robot actions and what each one does.

The language model never produces motor commands. It may place a *tag* such as ``[wave]`` in
its reply; this module maps that tag to a fixed list of Carle daemon step items taken from
Carle's hardware-verified movement and pose vocabulary (docs/movement-vocabulary.md,
docs/pose-vocabulary.md, docs/protocol-reference.md in the Carle checkout). The daemon's own
guard (one joint change per frame, 0.5 s minimum hold) still applies to everything here.

Carle code cheat-sheet used below
---------------------------------
``pose:N`` (0xB6, *holds*): 1/2 left arm forward up/down, 3/4 right arm forward up/down,
5/6 left shoulder lateral out/in, 7/8 right shoulder lateral out/in, 9/10 left elbow
up/down, 11/12 right elbow up/down.
``gesture:N`` (0xB2, fire once): 19 both arms down (home), 25 torso sway, 27 lean/bow.
``face:N`` (0xB2, held): 39 sad, 41 laughing, 43 concentrating, 45 talking, 47 big smile,
0 clears the hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# --- faces --------------------------------------------------------------------------------

FACE_SAD = 39
FACE_LAUGH = 41
FACE_THINKING = 43
FACE_TALKING = 45
FACE_HAPPY = 47
FACE_CLEAR = 0

#: Fire-once gesture that brings both arms home (Carle's own neutral return uses it too).
ARMS_HOME = 19


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


#: The LED face held in each conversational state.
STATE_FACES: dict[State, int] = {
    State.IDLE: FACE_CLEAR,
    State.LISTENING: FACE_HAPPY,
    State.THINKING: FACE_THINKING,
    State.SPEAKING: FACE_TALKING,
}


# --- gestures -----------------------------------------------------------------------------


def _pose(n: int, hold: float = 0.5) -> dict:
    return {"pose": n, "hold": hold}


def _gesture(n: int) -> dict:
    return {"gesture": n}


def _pause(seconds: float) -> dict:
    return {"pause": seconds}


@dataclass(frozen=True)
class Gesture:
    """One approved action: what the model may ask for and what the robot actually does."""

    tag: str
    description: str  # shown to the model so it picks sensibly
    steps: list[dict] = field(default_factory=list)  # daemon wire items, in order
    face: int | None = None  # a face to hold for the sentence, if the tag carries emotion
    duration: float = 0.0  # approximate seconds the arms are busy

    @property
    def moves(self) -> bool:
        return bool(self.steps)


def _sum_hold(steps: list[dict]) -> float:
    total = 0.0
    for s in steps:
        if "pose" in s:
            total += float(s.get("hold", 0.5))
        elif "pause" in s:
            total += float(s["pause"])
        elif "gesture" in s:
            total += 1.0  # canned animations take about a second
    return total


def _g(tag: str, description: str, steps: list[dict], face: int | None = None) -> Gesture:
    return Gesture(tag, description, steps, face, _sum_hold(steps))


# Right-arm wave: raise the right arm forward, bob the forearm twice, lower it.
_WAVE = [_pose(3), _pose(11), _pose(12), _pose(11), _pose(12), _pose(4)]
# Left-arm wave, mirror of the above — Carle's pose vocabulary calls this "Wave Goodbye".
_WAVE_LEFT = [_pose(1), _pose(9), _pose(10), _pose(9), _pose(10), _pose(2)]
# Both arms up to shoulder height, a little fist pump, then down.
_CELEBRATE = [_pose(1), _pose(3), _pose(9), _pose(11), _pose(10), _pose(12), _pose(2), _pose(4)]
# Arms out, forearms up — the palms-up shrug — held a beat, then home.
_SHRUG = [_pose(5), _pose(7), _pose(9), _pose(11), _pause(0.8), _pose(10), _pose(12), _pose(6), _pose(8)]
# Hand near the face ("The Thinker"), held, then down.
_THINK = [_pose(1), _pose(9), _pause(1.2), _pose(10), _pose(2)]
# A small forward motion of one arm, the kind of thing people do while explaining.
_EXPLAIN = [_pose(1), _pause(0.3), _pose(2)]
# Right arm extended forward — presenting, offering, pointing the way.
_PRESENT = [_pose(3), _pause(1.0), _pose(4)]
# Both arms wide — "ta-da" / open welcome.
_ARMS_WIDE = [_pose(5), _pose(7), _pause(1.0), _pose(6), _pose(8)]
# A quick double forward raise of the left arm reads as an emphatic nod/agreement.
_NOD = [_pose(1), _pose(2), _pose(1), _pose(2)]

GESTURES: dict[str, Gesture] = {
    g.tag: g
    for g in (
        _g("wave", "wave hello with the right arm (greetings)", _WAVE),
        _g("goodbye", "wave goodbye with the left arm (farewells)", _WAVE_LEFT),
        _g("celebrate", "both arms up in celebration (excitement, good news)", _CELEBRATE, FACE_HAPPY),
        _g("shrug", "a palms-up shrug (uncertainty, 'I don't know')", _SHRUG),
        _g("think", "hand near the face, thinking pose (pondering)", _THINK, FACE_THINKING),
        _g("explain", "a small arm movement while explaining a point", _EXPLAIN),
        _g("present", "extend the right arm forward (presenting, offering, pointing)", _PRESENT),
        _g("open", "both arms wide open (welcome, 'ta-da')", _ARMS_WIDE, FACE_HAPPY),
        _g("nod", "an emphatic double arm motion (agreement, yes)", _NOD),
        _g("sway", "a little torso dance sway (fun, music, playfulness)", [_gesture(25)], FACE_HAPPY),
        _g("bow", "a forward bow (thanks, respect, goodbye)", [_gesture(27)]),
        _g("happy", "a big smile, no arm movement", [], FACE_HAPPY),
        _g("laugh", "a laughing face, no arm movement", [], FACE_LAUGH),
        _g("sad", "a sad face, no arm movement", [], FACE_SAD),
    )
}

#: Tags the model is told about, in the order they are described in the system prompt.
TAG_NAMES: tuple[str, ...] = tuple(GESTURES)


def describe_for_prompt() -> str:
    """The approved-tag list in the form the system prompt embeds."""
    return "\n".join(f"[{g.tag}] — {g.description}" for g in GESTURES.values())


def steps_for(tag: str) -> list[dict]:
    """Daemon items for an approved tag, or an empty list for an unknown one (never raise)."""
    g = GESTURES.get(tag)
    return list(g.steps) if g else []


def face_for(tags: list[str], default: int) -> int:
    """The face to hold for a sentence: the last emotional tag wins, else ``default``."""
    face = default
    for t in tags:
        g = GESTURES.get(t)
        if g and g.face is not None:
            face = g.face
    return face


def face_item(code: int) -> dict:
    return {"face": code}


def home_items() -> list[dict]:
    """Return the arms to rest and release the held face."""
    return [face_item(FACE_CLEAR), _gesture(ARMS_HOME)]
