from companion import behaviour
from companion.dialogue import SentenceAssembler, split_sentences, strip_tags


def test_every_gesture_is_servo_safe_shape():
    # Each step is a single daemon item type with one joint at a time; holds never below 0.5 s.
    for g in behaviour.GESTURES.values():
        for s in g.steps:
            assert len([k for k in s if k in ("pose", "gesture", "pause", "face")]) == 1
            if "pose" in s:
                assert 1 <= s["pose"] <= 12
                assert s.get("hold", 0.5) >= 0.5
            if "gesture" in s:
                assert s["gesture"] in (19, 25, 27)


def test_unknown_tag_yields_no_steps():
    assert behaviour.steps_for("launch_missiles") == []
    assert behaviour.steps_for("wave")[0] == {"pose": 3, "hold": 0.5}


def test_face_for_uses_last_emotional_tag():
    assert behaviour.face_for(["wave"], behaviour.FACE_TALKING) == behaviour.FACE_TALKING
    assert behaviour.face_for(["sad", "happy"], behaviour.FACE_TALKING) == behaviour.FACE_HAPPY
    assert behaviour.face_for(["celebrate", "explain"], 45) == behaviour.FACE_HAPPY


def test_prompt_description_lists_every_tag():
    text = behaviour.describe_for_prompt()
    for tag in behaviour.TAG_NAMES:
        assert f"[{tag}]" in text


def test_strip_tags_cleans_text_and_filters_unknown():
    clean, tags = strip_tags("[wave] Hello there [unknown] , friend [happy] !")
    assert clean == "Hello there, friend!"
    assert tags == ["wave", "happy"]


def test_split_sentences_attaches_tags_to_their_sentence():
    s = split_sentences("[wave] Hi Elliott! I'm doing well. [celebrate] Great news: it works!")
    assert [x.text for x in s] == ["Hi Elliott!", "I'm doing well.", "Great news: it works!"]
    assert s[0].tags == ["wave"]
    assert s[1].tags == []
    assert s[2].tags == ["celebrate"]


def test_streaming_assembler_waits_for_open_tag_and_boundaries():
    a = SentenceAssembler()
    assert a.feed("Sure thing") == []
    first = a.feed(". Let me [thi")  # sentence 1 closes; the open tag belongs to sentence 2
    assert [x.text for x in first] == ["Sure thing."]
    assert a.feed("nk] about that") == []  # tag closed, but no sentence boundary yet
    out = a.feed(". The answer")
    assert [x.text for x in out] == ["Let me about that."]
    assert out[0].tags == ["think"]
    assert a.feed(" is 3.14 roughly") == []  # decimal point does not split
    rest = a.flush()
    assert [x.text for x in rest] == ["The answer is 3.14 roughly"]


def test_newline_splits_and_empty_pieces_are_dropped():
    s = split_sentences("First line\n\n[nod]\nSecond line")
    assert [x.text for x in s] == ["First line", "", "Second line"]
    assert s[1].tags == ["nod"] and not s[1].speakable
