"""Language-model providers, the conversation memory, and the system prompt."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from companion import behaviour
from companion.config import Settings

_log = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str


class LLM(Protocol):
    name: str

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]: ...


def build_system_prompt(settings: Settings) -> str:
    who = f" The person you are talking to is called {settings.user_name}." if settings.user_name else ""
    return f"""{settings.personality_text()}{who}

You are a physical robot standing in the room, not a chat window. Your replies are spoken aloud
through your own speaker, so:
- Talk like a person: short, natural sentences. Usually one to three sentences; go longer only
  when the question genuinely needs it.
- No markdown, no bullet points, no headings, no emoji, nothing that cannot be spoken.
- Remember what was said earlier in this conversation and resolve follow-ups from it.
- If you did not understand, say so briefly and ask.

You can move your arms and change your LED face. Do this by placing a tag in square brackets
right before the sentence it belongs to. Use at most two tags per reply, only when a human would
naturally gesture or show the feeling; most sentences need none. Never move continuously or for
no reason. Only these tags exist:
{behaviour.describe_for_prompt()}

Examples: "[wave] Hello! Good to see you." / "[think] Hmm, let me think about that." /
"[celebrate] That's brilliant news!" / "[goodbye] Bye for now, talk soon."
Never invent other tags and never describe your movements in words."""


class Conversation:
    """The running transcript the model sees, trimmed to a turn budget."""

    def __init__(self, max_turns: int = 40):
        self.max_turns = max_turns
        self.messages: list[Message] = []

    def add_user(self, text: str) -> None:
        self.messages.append(Message("user", text))
        self._trim()

    def add_assistant(self, text: str) -> None:
        if self.messages and self.messages[-1].role == "assistant":
            # Two assistant turns in a row (an interrupted reply then a new one) — merge so
            # providers that require alternation stay happy.
            self.messages[-1].content += "\n" + text
        else:
            self.messages.append(Message("assistant", text))
        self._trim()

    def reset(self) -> None:
        self.messages = []

    def _trim(self) -> None:
        excess = len(self.messages) - self.max_turns * 2
        if excess > 0:
            self.messages = self.messages[excess:]
        # Providers want the first message to be from the user.
        while self.messages and self.messages[0].role != "user":
            self.messages.pop(0)


class AnthropicLLM:
    def __init__(self, api_key: str, model: str = "claude-sonnet-5-5", max_tokens: int = 400):
        import anthropic  # noqa: PLC0415

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"Claude ({model})"

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        ) as stream:
            yield from stream.text_stream


class OpenAILLM:
    """OpenAI, or anything speaking its chat-completions API (Ollama, LM Studio, ...)."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str | None = None, max_tokens: int = 400):
        from openai import OpenAI  # noqa: PLC0415

        self._client = OpenAI(api_key=api_key or "not-needed", base_url=base_url or None)
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"{'OpenAI' if not base_url else base_url} ({model})"

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            stream=True,
            messages=[{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
        )
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class ScriptedLLM:
    """No API at all: canned replies so the whole pipeline can be exercised offline."""

    name = "offline demo (no AI)"

    _RULES: list[tuple[re.Pattern, str]] = [
        (re.compile(r"\b(hello|hi|hey|good (morning|afternoon|evening))\b", re.I),
         "[wave] Hello there! I'm running in offline demo mode, so my answers are canned, "
         "but my ears, voice and arms are all real."),
        (re.compile(r"\b(bye|goodbye|see you)\b", re.I),
         "[goodbye] Goodbye! It was lovely talking with you."),
        (re.compile(r"\b(your name|who are you)\b", re.I),
         "[present] I'm Ruko, a little robot with big ambitions. "),
        (re.compile(r"\b(dance|party|music)\b", re.I),
         "[sway] Ooh, I love a bit of a dance. [happy] How's that?"),
        (re.compile(r"\b(thank|thanks)\b", re.I), "[bow] You're very welcome."),
        (re.compile(r"\b(joke|funny)\b", re.I),
         "Why did the robot go on holiday? [laugh] It needed to recharge its batteries."),
        (re.compile(r"\b(how are you|how do you feel)\b", re.I),
         "[happy] I'm feeling fully charged, thanks for asking. How about you?"),
    ]

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:  # noqa: ARG002
        text = messages[-1].content if messages else ""
        for pattern, reply in self._RULES:
            if pattern.search(text):
                yield reply
                return
        yield (
            f"[shrug] You said: {text.strip() or 'nothing I could hear'}. "
            "I'm in offline demo mode, so add an API key in settings to make me actually clever."
        )


def make_llm(settings: Settings, secrets: dict[str, str]) -> LLM:
    p = settings.llm_provider
    if p == "anthropic":
        key = secrets.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Claude selected but ANTHROPIC_API_KEY is not set")
        return AnthropicLLM(key, settings.llm_model or "claude-sonnet-5-5", settings.max_reply_tokens)
    if p == "openai":
        key = secrets.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OpenAI selected but OPENAI_API_KEY is not set")
        return OpenAILLM(key, settings.llm_model or "gpt-4o-mini", None, settings.max_reply_tokens)
    if p == "openai_compatible":
        if not settings.llm_base_url:
            raise RuntimeError("openai_compatible provider needs llm_base_url (e.g. http://localhost:11434/v1)")
        return OpenAILLM(secrets.get("OPENAI_API_KEY", ""), settings.llm_model, settings.llm_base_url, settings.max_reply_tokens)
    return ScriptedLLM()
