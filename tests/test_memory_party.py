from datetime import date, datetime

import numpy as np

from companion import memory
from companion.context import birthday_text, mood_text
from companion.llm import Message


class FakeLLM:
    name = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def stream(self, system, messages):
        self.prompts.append(messages[-1].content)
        yield self.reply


def test_memory_add_dedupes_dates_and_caps(tmp_path):
    lines = memory.add([], ["Augie loves Grave Digger", "- Augie loves grave digger!", ""], date(2026, 11, 22))
    assert lines == ["(22 Nov 2026) Augie loves Grave Digger"]
    lines = memory.add(lines, ["Pancake the chicken laid an egg"], date(2026, 11, 23))
    assert lines[-1].startswith("(23 Nov 2026) Pancake")
    big = memory.add(lines, [f"fact {i}" for i in range(100)])
    assert len(big) == memory.MAX_LINES
    p = tmp_path / "m.md"
    memory.save(lines, p)
    assert memory.load(p) == lines
    assert "Grave Digger" in memory.for_prompt(lines)


def test_extract_parses_list_and_none():
    llm = FakeLLM("- Augie's favourite truck is Grave Digger\n- We danced to Baby Shark\n")
    out = memory.extract(llm, "Augie", [Message("user", "I like Grave Digger"), Message("assistant", "[happy] Cool!")], [])
    assert out == ["Augie's favourite truck is Grave Digger", "We danced to Baby Shark"]
    assert "Augie: I like Grave Digger" in llm.prompts[0] and "[happy]" not in llm.prompts[0]
    assert memory.extract(FakeLLM("NONE"), "Augie", [Message("user", "hi"), Message("assistant", "hi")], []) == []


def test_merge_falls_back_when_model_reply_is_odd():
    lines = [f"({i} Jan 2026) fact {i}" for i in range(50)]
    assert memory.merge(FakeLLM("nonsense"), lines) == lines[-memory.MERGE_TARGET:]
    good = "\n".join(f"- merged {i}" for i in range(10))
    assert memory.merge(FakeLLM(good), lines) == [f"merged {i}" for i in range(10)]


def test_birthday_and_mood_context():
    assert birthday_text("Augie", "2023-05-14", date(2026, 5, 14)) == (
        "Today is Augie's birthday! Augie is 3 today. Make a big happy fuss and offer a birthday song."
    )
    assert birthday_text("Augie", "2023-05-14", date(2026, 5, 11)) == "Augie's birthday is in 3 days (Augie will be 3)."
    assert birthday_text("Augie", "2023-05-14", date(2026, 6, 1)) == ""
    assert birthday_text("Augie", "nonsense", date(2026, 6, 1)) == ""
    assert "bedtime" in mood_text(datetime(2026, 1, 1, 19, 30))
    assert mood_text(datetime(2026, 1, 1, 10, 0)) == ""


def test_dance_party_plays_every_song_until_interrupted(monkeypatch, tmp_path):
    from companion import conversation
    from companion.config import Settings
    from companion.conversation import Companion
    from companion.music import Song

    played = []
    c = Companion(Settings(hands_free=False, music_dir=str(tmp_path)), {}, lambda e, d: None)
    c.running = True
    c.library.songs = [Song("A", tmp_path / "a"), Song("B", tmp_path / "b"), Song("C", tmp_path / "c")]
    monkeypatch.setattr(conversation.time, "sleep", lambda s: None)
    c._dance = lambda title: played.append(title)
    c._dance_party()
    assert sorted(played) == ["A", "B", "C"]
    played.clear()

    def dance_then_interrupt(title):
        played.append(title)
        c._party_cancel.set()

    c._dance = dance_then_interrupt
    c._dance_party()
    assert len(played) == 1
