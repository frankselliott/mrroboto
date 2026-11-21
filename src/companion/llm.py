"""Language-model providers, the conversation memory, and the system prompt."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
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
  when the question genuinely needs it. Make the first sentence short, since you start
  speaking as soon as it is ready.
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


class ClaudeCodeLLM:
    """Claude through the Claude Code CLI's headless mode — uses the signed-in Pro/Max account.

    Each turn runs ``claude -p`` with the companion's system prompt, streaming text deltas
    back. Conversation memory is Claude Code's own session: the first turn creates it with
    ``--session-id`` and later turns ``--resume`` it, so follow-ups work without re-sending
    history. Slower to first word than the API (the CLI starts a fresh process each time) and
    subject to the subscription's rate limits, but costs no API tokens.
    """

    def __init__(self, model: str = "sonnet", binary: str | None = None):
        self.binary = binary or find_claude_cli()
        if not self.binary:
            raise RuntimeError("Claude Code CLI not found — install it or sign in to the Claude app")
        self.model = model or "sonnet"
        self.name = f"Claude Code ({self.model}, your subscription)"
        self._session: str | None = None

    def new_session(self) -> None:
        self._session = None

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        import json  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        import uuid  # noqa: PLC0415

        prompt = messages[-1].content if messages else ""
        cmd = [
            self.binary, "-p", prompt,
            "--output-format", "stream-json", "--verbose", "--include-partial-messages",
            "--model", self.model, "--system-prompt", system,
            "--disallowedTools", "Bash,Read,Edit,Write,Glob,Grep,WebFetch,WebSearch,Agent,Task,NotebookEdit,TodoWrite",
            "--max-turns", "1",
            "--strict-mcp-config",  # skip the user's MCP servers: faster start, no side effects
        ]
        if self._session:
            cmd += ["--resume", self._session]
        else:
            self._session = str(uuid.uuid4())
            cmd += ["--session-id", self._session]
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
        env["CLAUDE_CODE_ENTRYPOINT"] = "ruko-companion"
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
            cwd=str(Path.home()),
        )
        streamed = False
        final_text = ""
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = ev.get("type")
                if t == "stream_event":
                    inner = ev.get("event", {})
                    if inner.get("type") == "content_block_delta":
                        delta = inner.get("delta", {})
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            streamed = True
                            yield delta["text"]
                elif t == "result":
                    final_text = ev.get("result") or ""
                    if ev.get("is_error"):
                        raise RuntimeError(final_text or "Claude Code returned an error")
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.terminate()
        if not streamed and final_text:
            yield final_text
        if proc.returncode not in (0, None) and not streamed and not final_text:
            err = (proc.stderr.read() if proc.stderr else "")[-400:]
            raise RuntimeError(f"Claude Code failed ({proc.returncode}): {err.strip() or 'not logged in?'}")


def find_claude_cli() -> str | None:
    """The `claude` binary: PATH, the usual install spots, or the one bundled in Claude.app."""
    import glob  # noqa: PLC0415
    import shutil  # noqa: PLC0415

    for c in (shutil.which("claude"), Path.home() / ".local/bin/claude", Path.home() / ".claude/local/claude"):
        if c and Path(c).exists():
            return str(c)
    bundled = sorted(
        glob.glob(
            str(Path.home() / "Library/Application Support/Claude/claude-code/*/*/claude.app/Contents/MacOS/claude")
        )
    )
    return bundled[-1] if bundled else None


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
    if p == "claude_code":
        return ClaudeCodeLLM(settings.llm_model or "sonnet")
    if p == "openai_compatible":
        if not settings.llm_base_url:
            raise RuntimeError("openai_compatible provider needs llm_base_url (e.g. http://localhost:11434/v1)")
        return OpenAILLM(secrets.get("OPENAI_API_KEY", ""), settings.llm_model, settings.llm_base_url, settings.max_reply_tokens)
    return ScriptedLLM()
