import os
import stat

from companion import config
from companion.llm import Conversation, ScriptedLLM, build_system_prompt


def test_settings_roundtrip_and_coercion(tmp_path):
    p = tmp_path / "config.yaml"
    s = config.Settings(robot_address="ABC", tts_speed=1.2, hands_free=True)
    config.save_settings(s, p)
    loaded = config.load_settings(p)
    assert loaded == s
    # Strings from a form are coerced to the field types; unknown keys are ignored.
    s2 = config.Settings.from_dict({"tts_speed": "0.8", "hands_free": "false", "max_history_turns": "12", "bogus": 1})
    assert s2.tts_speed == 0.8 and s2.hands_free is False and s2.max_history_turns == 12


def test_secrets_file_is_owner_only_and_never_in_project(tmp_path):
    p = tmp_path / "secrets.env"
    config.set_secret("ANTHROPIC_API_KEY", "sk-test-123", p)
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    assert config.load_secrets(p)["ANTHROPIC_API_KEY"] == "sk-test-123"
    config.set_secret("ANTHROPIC_API_KEY", "", p)
    assert "ANTHROPIC_API_KEY" not in config.load_secrets(p)
    assert config.secret_status({"OPENAI_API_KEY": "x"}) == {
        "ANTHROPIC_API_KEY": False, "OPENAI_API_KEY": True, "ELEVENLABS_API_KEY": False,
    }
    assert not config.SECRETS_PATH.is_relative_to(config.ROOT)


def test_conversation_trims_and_keeps_alternation():
    c = Conversation(max_turns=2)
    for i in range(5):
        c.add_user(f"q{i}")
        c.add_assistant(f"a{i}")
    assert [m.content for m in c.messages] == ["q3", "a3", "q4", "a4"]
    assert c.messages[0].role == "user"
    # An interrupted reply followed by another reply merges rather than breaking alternation.
    c.add_assistant("extra")
    assert c.messages[-1].content == "a4\nextra"


def test_system_prompt_mentions_tags_and_personality():
    s = config.Settings(personality="witty", user_name="Elliott")
    prompt = build_system_prompt(s)
    assert "[wave]" in prompt and "Elliott" in prompt and "dry, quick wit" in prompt
    s.personality, s.personality_custom = "custom", "You are a pirate robot."
    assert "pirate" in build_system_prompt(s)


def test_scripted_llm_only_uses_approved_tags():
    from companion.dialogue import strip_tags

    llm = ScriptedLLM()
    for text in ("hello", "goodbye", "dance", "thanks", "joke", "how are you", "what is 2+2"):
        reply = "".join(llm.stream("", [__import__("companion.llm", fromlist=["Message"]).Message("user", text)]))
        clean, tags = strip_tags(reply)
        assert clean
        assert "[" not in clean
