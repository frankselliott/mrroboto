"""Language-model providers, the conversation memory, and the system prompt."""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
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


def build_system_prompt(settings: Settings, song_titles: list[str] | None = None, memories: str = "") -> str:
    who = f" The person you are talking to is usually called {settings.user_name}." if settings.user_name else ""
    owner = ""
    if settings.owner_name:
        owner = (
            f"\n\nYou are {settings.owner_name}'s robot: you belong to {settings.owner_name}, you are proud "
            f"of it, and you say so if asked who you belong to."
        )
        if settings.owner_profile.strip():
            owner += f" Everything you know about {settings.owner_name}:\n{settings.owner_profile.strip()}"
    if memories:
        owner += (
            f"\n\nThings you remember from earlier days (oldest first; bring one up when it fits, "
            f"like a friend would, but don't list them):\n{memories}"
        )
    songs = ""
    if song_titles:
        listing = "\n".join(f"- {t}" for t in song_titles[:40])
        songs = f"""

You can play music and dance to it. When asked to play a song, dance, or when a party is
clearly wanted, reply with one short sentence and put the tag [dance:<exact title>] at the END
of that sentence, using a title from this list exactly as written:
{listing}
If the request does not match a title, say which songs you do have instead. Never invent titles.
For a dance party (several songs in a row) use [dance:party]. While the music plays you dance on
your own; the person can stop you by talking to you."""
    persona = settings.personality_text()
    if not persona.lower().startswith("you are"):
        persona = f"You are {settings.robot_name}, a small robot companion with LED eyes and two little arms. {persona}"
    return f"""{persona}{who}{owner}

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
Never invent other tags and never describe your movements in words.{songs}"""


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
    """Claude through the Claude Code CLI — uses the signed-in Pro/Max account, no API key.

    One ``claude -p`` process is kept alive and fed user messages over stdin
    (``--input-format stream-json``), so after the first reply the next word arrives in about
    a second rather than the five-plus seconds a fresh process costs. Memory is Claude Code's
    own session: if the process has to be restarted (an interruption kills it, since it would
    otherwise keep generating) the same session is ``--resume``d. Tools and MCP servers are
    disabled; the companion's system prompt replaces Claude Code's.
    """

    _NO_TOOLS = "Bash,Read,Edit,Write,Glob,Grep,WebFetch,WebSearch,Agent,Task,NotebookEdit,TodoWrite"

    def __init__(self, model: str = "sonnet", binary: str | None = None):
        self.binary = binary or find_claude_cli()
        if not self.binary:
            raise RuntimeError("Claude Code CLI not found — install it or sign in to the Claude app")
        self.model = model or "sonnet"
        self.name = f"Claude Code ({self.model}, your subscription)"
        self._session: str | None = None
        self._proc = None
        self._system: str | None = None
        self._events: queue.Queue = queue.Queue()
        self._lock = threading.Lock()

    # --- process management ----------------------------------------------------------------

    def _start(self, system: str) -> None:
        import subprocess  # noqa: PLC0415
        import uuid  # noqa: PLC0415

        self.close()
        cmd = [
            self.binary, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
            "--verbose", "--include-partial-messages", "--model", self.model,
            "--system-prompt", system, "--disallowedTools", self._NO_TOOLS,
            "--max-turns", "1", "--strict-mcp-config",
        ]
        if self._session:
            cmd += ["--resume", self._session]
        else:
            self._session = str(uuid.uuid4())
            cmd += ["--session-id", self._session]
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
        env["CLAUDE_CODE_ENTRYPOINT"] = "ruko-companion"
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, env=env, cwd=str(Path.home()),
        )
        self._system = system
        self._events = queue.Queue()
        threading.Thread(target=self._reader, args=(self._proc, self._events), daemon=True).start()
        _log.info("claude code process started (pid %d, session %s)", self._proc.pid, self._session[:8])

    @staticmethod
    def _reader(proc, events: queue.Queue) -> None:
        import json  # noqa: PLC0415

        for line in proc.stdout:
            if line.startswith("{"):
                try:
                    events.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
        events.put(None)  # EOF: the process is gone

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                proc.kill()

    def new_session(self) -> None:
        self.close()
        self._session = None

    # --- the provider interface -------------------------------------------------------------

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        import json  # noqa: PLC0415

        with self._lock:
            if self._proc is None or self._proc.poll() is not None or system != self._system:
                self._start(system)
            while not self._events.empty():  # stale events from an earlier turn
                self._events.get_nowait()
            prompt = messages[-1].content if messages else ""
            try:
                self._proc.stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": prompt}}) + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError) as exc:
                self.close()
                raise RuntimeError(f"Claude Code process unavailable: {exc}") from exc
            try:
                yield from self._consume()
            except GeneratorExit:
                # The caller stopped early (an interruption). The process would keep generating
                # into the void, so drop it; the session is resumed on the next turn.
                self.close()
                raise

    def _consume(self) -> Iterator[str]:
        """Yield the text of the reply to the message just written.

        Events from the previous reply can still be arriving (its ``result`` bookkeeping lands
        after its ``message_stop``), so nothing counts until this reply's ``message_start``.
        """
        started = False
        streamed = False
        while True:
            try:
                ev = self._events.get(timeout=120)
            except queue.Empty as exc:
                self.close()
                raise RuntimeError("Claude Code did not answer within 120 s") from exc
            if ev is None:
                self.close()
                raise RuntimeError("Claude Code exited — are you signed in? (run: claude auth login)")
            t = ev.get("type")
            if t == "stream_event":
                inner = ev.get("event", {})
                kind = inner.get("type")
                if kind == "message_start":
                    started = True
                elif kind == "content_block_delta" and started:
                    delta = inner.get("delta", {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        streamed = True
                        yield delta["text"]
                elif kind == "message_stop" and started:
                    return  # the text is complete; the result bookkeeping arrives later
            elif t == "result":
                if ev.get("is_error"):
                    raise RuntimeError(ev.get("result") or "Claude Code returned an error")
                if started:
                    if not streamed and ev.get("result"):
                        yield ev["result"]
                    return
                # else: the previous reply's result — ignore it


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
