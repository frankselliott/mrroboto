"""Run a whole dance session against fakes: routines must be queued in order, on time."""

import time

import numpy as np

from companion import conversation
from companion.behaviour import State
from companion.config import Settings
from companion.conversation import Companion
from companion.music import Song


class FakeDaemon:
    def __init__(self):
        self.items = []

    def enqueue(self, items):
        self.items.append(list(items))
        return len(items)

    def clear(self):
        pass

    def stop(self):
        pass


class FakeSpeak:
    def __init__(self):
        self.bytes = 0
        self.t0 = None

    def stream_pcm(self, chunks, samplerate, channels=1):
        self.t0 = time.monotonic()
        for b in chunks:
            self.bytes += len(b)
        return {"ok": True, "outcome": "completed"}

    def stop(self):
        return True


def test_dance_session_queues_routines_on_the_beat(monkeypatch, tmp_path):
    sr = 24000
    song = np.zeros(sr * 5, dtype=np.int16)  # 5 s of "music"
    monkeypatch.setattr(conversation, "decode", lambda s: song)
    monkeypatch.setattr(conversation, "estimate_bpm", lambda pcm, sr: 120.0)
    events = []
    c = Companion(Settings(hands_free=False, music_dir=str(tmp_path)), {}, lambda e, d: events.append((e, d)))
    c.running = True
    c.daemon = FakeDaemon()
    c.speak = FakeSpeak()
    c.library.songs = [Song("Test Song", tmp_path / "test.mp3")]
    c.last_status = {"robot": {"connected": True}}
    t0 = time.monotonic()
    c._dance("test song")
    elapsed = time.monotonic() - t0
    assert 4.5 < elapsed < 8, elapsed  # real-time paced, not instant
    assert c.speak.bytes == len(song) * 2
    routines = [d["tag"] for e, d in events if e == "gesture"]
    assert len(routines) >= 1 and all(t.startswith("dance: ") for t in routines)
    # Each queued batch is a full routine (several servo-safe steps), then the arms go home.
    batches = [b for b in c.daemon.items if any("pose" in s or "gesture" in s or "waist" in s for s in b)]
    assert len(batches[0]) >= 5
    assert c.daemon.items[-1] == [{"face": 0}, {"gesture": 19}]
    assert c.state == State.IDLE
    assert any(e == "notice" and "beats per minute" in d["message"] for e, d in events)
