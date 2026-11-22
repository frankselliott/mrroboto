"""The hands-free wake-word flow, with every external piece stubbed out."""

import numpy as np

from companion.behaviour import State
from companion.config import Settings
from companion.conversation import Companion


class FakeSTT:
    name = "fake"

    def __init__(self, text):
        self.text = text

    def transcribe(self, audio, sr):
        return self.text

    def warm_up(self):
        pass


def make(text, hands_free=True, wake=True):
    events = []
    s = Settings(hands_free=hands_free, wake_word_enabled=wake, wake_word="Mr Roboto", music_dir="/nonexistent")
    c = Companion(s, {}, lambda e, d: events.append((e, d)))
    c.running = True
    c.stt = FakeSTT(text)
    turns = []
    c._run_turn = lambda t: turns.append(t)
    said = []
    c._say_short = lambda t: said.append(t)
    c._apply_face = lambda code, force=False: None
    return c, events, turns, said


def test_ignores_speech_without_wake_word():
    c, events, turns, said = make("what time is it")
    c._handle_utterance(np.ones(16000, dtype=np.float32) * 0.1, hands_free=True)
    assert turns == [] and said == []
    assert c.state == State.LISTENING


def test_wake_word_with_question_runs_turn():
    c, events, turns, said = make("Mister Roboto, what time is it?")
    c._handle_utterance(np.ones(16000, dtype=np.float32) * 0.1, hands_free=True)
    assert turns == ["what time is it?"]
    assert ("transcript", {"role": "user", "text": "what time is it?"}) in events


def test_bare_wake_word_acknowledges_then_takes_next_utterance():
    c, events, turns, said = make("Mr Roboto")
    c._handle_utterance(np.ones(16000, dtype=np.float32) * 0.1, hands_free=True)
    assert said == ["Yes?"] and turns == []
    c.stt = FakeSTT("tell me a joke")
    c._handle_utterance(np.ones(16000, dtype=np.float32) * 0.1, hands_free=True)
    assert turns == ["tell me a joke"]


def test_push_to_talk_never_needs_wake_word():
    c, events, turns, said = make("tell me a joke")
    c._handle_utterance(np.ones(16000, dtype=np.float32) * 0.1, hands_free=False)
    assert turns == ["tell me a joke"]


def test_silent_mic_is_reported_not_transcribed():
    c, events, turns, said = make("anything")
    c._handle_utterance(np.zeros(16000, dtype=np.float32), hands_free=False)
    assert turns == []
    assert any(e == "error" and "silence" in d["message"] for e, d in events)


def test_pronunciation_substitution():
    c, *_ = make("x")
    assert c._pronounce("Hi Augie, Augie's robot here.") == "Hi Auggie, Auggie's robot here."


def test_follow_up_window_lets_next_utterance_skip_wake_word():
    c, events, turns, said = make("what about the chickens?")
    c.settings.follow_up_seconds = 15
    c._open_follow_up_window()
    c._handle_utterance(np.ones(16000, dtype=np.float32) * 0.1, hands_free=True)
    assert turns == ["what about the chickens?"]
    # the window closes once used
    c.stt = FakeSTT("and the cars?")
    c._handle_utterance(np.ones(16000, dtype=np.float32) * 0.1, hands_free=True)
    assert turns == ["what about the chickens?"]


def test_idle_nudge_only_when_ready_and_limited():
    c, events, turns, said = make("x")
    initiated = []
    c._initiate = lambda kind: initiated.append(kind)
    c._tts_ready = True
    c.llm = object()  # any provider; never called here
    c.settings.idle_nudge_minutes = 0.0001
    c.settings.idle_nudge_max = 2
    c._last_interaction -= 60
    ready = {"speaker": True, "robot": {"connected": True}}
    c._maybe_nudge({"speaker": False, "robot": {"connected": True}})
    assert initiated == []  # speaker not ready
    c._maybe_nudge(ready); c._last_interaction -= 60; c._maybe_nudge(ready); c._last_interaction -= 60; c._maybe_nudge(ready)
    assert initiated == ["nudge", "nudge"]  # capped per silence
    c._note_interaction(); c._last_interaction -= 60; c._maybe_nudge(ready)
    assert initiated == ["nudge", "nudge", "nudge"]  # reset by a real interaction
