"""Turning a streamed model reply into speakable sentences with their gesture tags.

The model writes prose with inline tags like ``[wave]``. We need, as the text streams in:

- the spoken text with every tag removed (and tidy whitespace), and
- for each sentence, the approved tags that appeared in it, so the behaviour engine can fire
  them when that sentence starts playing.

Unknown tags are dropped silently: the model can only ever reach the approved list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from companion.behaviour import GESTURES

_TAG_RE = re.compile(r"\[([a-zA-Z_]+)(?::\s*([^\]]+?)\s*)?\]")
# A sentence ends at . ! ? (optionally followed by closing quotes/brackets) and whitespace,
# or at a newline. We deliberately do not split on abbreviations' dots mid-sentence when no
# whitespace follows, e.g. "3.14" or "e.g.".
_SENTENCE_END_RE = re.compile(r'([.!?]+["\')\]]*)(\s+|$)|\n+')


@dataclass
class Sentence:
    text: str
    tags: list[str] = field(default_factory=list)
    actions: list[tuple[str, str]] = field(default_factory=list)  # parametrised tags, e.g. ("dance", "Happy")

    @property
    def speakable(self) -> bool:
        return bool(re.search(r"\w", self.text))


ACTION_TAGS = ("dance", "spotify")


def strip_tags(text: str) -> tuple[str, list[str]]:
    """Remove every ``[tag]``; return the clean text and the approved tags found, in order."""
    clean, tags, _actions = strip_all(text)
    return clean, tags


def strip_all(text: str) -> tuple[str, list[str], list[tuple[str, str]]]:
    """Like ``strip_tags`` but also returns parametrised action tags such as ``[dance:Happy]``."""
    tags: list[str] = []
    actions: list[tuple[str, str]] = []
    for m in _TAG_RE.finditer(text):
        name = m.group(1).lower()
        arg = (m.group(2) or "").strip()
        if name in ACTION_TAGS and arg:
            actions.append((name, arg))
        elif name in GESTURES:
            tags.append(name)
    clean = _TAG_RE.sub(" ", text)
    clean = re.sub(r"[ \t]+", " ", clean)
    clean = re.sub(r" +([,.!?;:])", r"\1", clean).strip()
    return clean, tags, actions


class SentenceAssembler:
    """Feed streamed text chunks in; get complete sentences out as soon as they are complete.

    Tags are collected with the sentence they occur in. A tag that arrives at the very end
    of a sentence boundary is attached to the sentence just closed if it precedes the
    whitespace, otherwise to the next one.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: str) -> list[Sentence]:
        self._buf += chunk
        out: list[Sentence] = []
        while True:
            m = _SENTENCE_END_RE.search(self._buf)
            if m is None:
                break
            # Don't split while a tag is still open at the cut point: "[wa" could be "[wave]".
            end = m.end()
            if self._buf.count("[", 0, end) > self._buf.count("]", 0, end):
                break
            piece, self._buf = self._buf[:end], self._buf[end:]
            s = self._to_sentence(piece)
            if s is not None:
                out.append(s)
        return out

    def flush(self) -> list[Sentence]:
        piece, self._buf = self._buf, ""
        s = self._to_sentence(piece)
        return [s] if s is not None else []

    @staticmethod
    def _to_sentence(piece: str) -> Sentence | None:
        text, tags, actions = strip_all(piece)
        if not text and not tags and not actions:
            return None
        return Sentence(text, tags, actions)


def split_sentences(text: str) -> list[Sentence]:
    """One-shot convenience for a complete reply."""
    a = SentenceAssembler()
    return a.feed(text) + a.flush()
