"""Wake-word detection on transcribed text.

Hands-free mode transcribes every utterance the voice gate hears; this decides whether it was
addressed to the robot. Matching is forgiving about how Whisper spells the wake phrase
("Mr Roboto" comes back as "Mr. Roboto", "Mister Roboto", "Mr Robotto", "Mr Robot"...).
"""

from __future__ import annotations

import re

#: Common speech-to-text renderings of wake-phrase words.
_VARIANTS: dict[str, list[str]] = {
    "mr": ["mr", "mister", "missus", "mrs", "mista", "mr\\."],
    "roboto": ["roboto", "robotto", "robot o", "robot", "roberto", "rebotto", "robota", "ruboto"],
    "hey": ["hey", "hi", "hello", "ok", "okay"],
    "ruko": ["ruko", "rucko", "rouko", "roocoo", "ruco"],
}

_LEAD = r"^\W*(?:(?:hey|hi|hello|ok|okay)\W+)?"


def compile_wake(wake_word: str) -> re.Pattern:
    words = [w for w in re.split(r"[\s.]+", wake_word.lower()) if w]
    parts = []
    for w in words:
        alts = _VARIANTS.get(w, [re.escape(w)])
        parts.append("(?:" + "|".join(alts) + ")")
    body = r"\W*".join(parts)
    return re.compile(_LEAD + body + r"\b[\s,.!?:;]*", re.IGNORECASE)


def split_wake(text: str, wake_word: str) -> tuple[bool, str]:
    """(was the wake phrase at the start?, the rest of the utterance)."""
    if not text:
        return False, ""
    m = compile_wake(wake_word).match(text.strip())
    if not m:
        return False, text.strip()
    rest = text.strip()[m.end():].strip()
    return True, rest
