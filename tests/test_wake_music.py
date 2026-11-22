from pathlib import Path

from companion.config import Settings
from companion.dialogue import split_sentences, strip_all
from companion.music import Library, Song, clean_title
from companion.wakeword import split_wake


def test_wake_word_variants_and_remainder():
    for text in ("Mr Roboto, what's the weather?", "Mister Roboto what's the weather", "hey mr. robotto, what's the weather?", "Mr Robot, what's the weather?"):
        ok, rest = split_wake(text, "Mr Roboto")
        assert ok, text
        assert rest.lower().startswith("what's the weather")
    ok, rest = split_wake("Mr Roboto", "Mr Roboto")
    assert ok and rest == ""
    ok, rest = split_wake("What's the weather, Mr Roboto?", "Mr Roboto")
    assert not ok and rest.startswith("What's")
    assert split_wake("Hey Ruko, hello", "Ruko") == (True, "hello")


def test_action_tags_are_parsed_separately_from_gestures():
    clean, tags, actions = strip_all("[happy] Let's dance! [dance: Pharrell Williams - Happy]")
    assert clean == "Let's dance!"
    assert tags == ["happy"] and actions == [("dance", "Pharrell Williams - Happy")]
    s = split_sentences("Sure. [dance:Wheels on the Bus] Here we go!")
    assert s[1].actions == [("dance", "Wheels on the Bus")]
    assert strip_all("[dance] no argument is ignored")[2] == []


def test_library_matching(tmp_path: Path):
    for name in ("Pharrell Williams - Happy (Official Video).mp3", "Wheels on the Bus  @CoComelon Nursery Rhymes & Kids Songs.mp3", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    lib = Library(tmp_path)
    assert lib.titles() == ["Pharrell Williams - Happy", "Wheels on the Bus"]
    assert lib.find("happy").title == "Pharrell Williams - Happy"
    assert lib.find("the wheels on the bus").title == "Wheels on the Bus"
    assert lib.find("bus song").title == "Wheels on the Bus"
    assert lib.find("bohemian rhapsody") is None
    assert clean_title("Song (Official Audio)") == "Song"
    assert Song("A", tmp_path).key == "a"


def test_pronunciation_map_and_owner_defaults():
    s = Settings()
    assert s.robot_name == "Mr Roboto" and s.owner_name == "Augie"
    assert s.pronunciation_map() == {"Augie": "Auggie"}
    s.pronunciations = "Augie=Auggie\nElliott = Elliot\nbroken line"
    assert s.pronunciation_map() == {"Augie": "Auggie", "Elliott": "Elliot"}


def test_library_prefers_word_overlap_over_fuzzy(tmp_path: Path):
    for name in ("The-Robot-Song.m4a", "Tyrannosaurus-Rex-Dinosaur-Song-Pinkfong.m4a", "Styx-Mr-Roboto.m4a"):
        (tmp_path / name).write_bytes(b"")
    lib = Library(tmp_path)
    assert lib.find("the dinosaur song").title.startswith("Tyrannosaurus")
    assert lib.find("mr roboto").title == "Styx Mr Roboto"
    assert lib.find("robot song").title == "The Robot Song"
