"""Memory across days: short facts distilled from conversations, kept in a text file.

The file is ``memories.md`` next to the settings, one dated line per memory, readable and
editable by a parent. Facts are extracted by the language model from the recent transcript,
deduplicated, and capped; when the list grows long it is merged down by the model as well.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

from companion.config import APP_DIR
from companion.llm import LLM, Message

_log = logging.getLogger(__name__)

MEMORY_PATH = APP_DIR / "memories.md"
MAX_LINES = 60
MERGE_TARGET = 30


def load(path: Path = MEMORY_PATH) -> list[str]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("- "):
            out.append(line[2:].strip())
        elif line and not line.startswith("#"):
            out.append(line)
    return out


def save(lines: list[str], path: Path = MEMORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "# Mr Roboto's memories\n\n" + "".join(f"- {line}\n" for line in lines)
    path.write_text(body)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def add(lines: list[str], new: list[str], today: date | None = None) -> list[str]:
    """Append dated memories that are not already present (loosely); cap the list."""
    today = today or date.today()
    stamp = today.strftime("%-d %b %Y")
    existing = {_norm(re.sub(r"^\(.*?\)\s*", "", l)) for l in lines}
    out = list(lines)
    for item in new:
        item = item.strip().lstrip("-•* ").strip()
        if not item or _norm(item) in existing or len(item) > 200:
            continue
        existing.add(_norm(item))
        out.append(f"({stamp}) {item}")
    return out[-MAX_LINES:]


EXTRACT_PROMPT = """You help a child's robot companion remember things between days.
Below is a recent conversation between the child ({owner}) and the robot. List the facts worth
remembering on another day: things {owner} said about himself (likes, dislikes, names of toys,
friends, pets, places, events), things that happened (a dance, a story he liked, a game), and
requests for next time. Write each as one short plain sentence in the past or present tense,
one per line, starting with "- ". Do not repeat anything in the "already remembered" list, do
not invent, do not include the robot's own opinions, and skip small talk. If there is nothing
new worth keeping, reply with exactly: NONE

Already remembered:
{existing}

Conversation:
{transcript}"""

MERGE_PROMPT = """These are a robot companion's memories about a child, one per line, oldest first.
Merge and shorten them into at most {target} lines, keeping every distinct fact, preferring the
most recent version where facts changed, keeping the dates in parentheses where present (use the
latest date for merged lines). One memory per line, starting with "- ". Reply with the list only.

{lines}"""


def _parse_list(text: str) -> list[str]:
    if "NONE" in text.strip().upper()[:8]:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("- ", "• ", "* ")):
            out.append(line[2:].strip())
    return out


def extract(llm: LLM, owner: str, messages: list[Message], existing: list[str]) -> list[str]:
    transcript = "\n".join(
        f"{'Robot' if m.role == 'assistant' else owner}: {re.sub(r'\[[a-z_]+(?::[^\]]*)?\]', '', m.content).strip()}"
        for m in messages
    )
    prompt = EXTRACT_PROMPT.format(
        owner=owner or "the child", existing="\n".join(f"- {l}" for l in existing[-25:]) or "(nothing yet)", transcript=transcript
    )
    text = "".join(llm.stream("You are a careful note-taker. Reply only with the list.", [Message("user", prompt)]))
    return _parse_list(text)


def merge(llm: LLM, lines: list[str]) -> list[str]:
    prompt = MERGE_PROMPT.format(target=MERGE_TARGET, lines="\n".join(f"- {l}" for l in lines))
    text = "".join(llm.stream("You are a careful note-taker. Reply only with the list.", [Message("user", prompt)]))
    merged = _parse_list(text)
    return merged if 3 <= len(merged) <= MERGE_TARGET + 5 else lines[-MERGE_TARGET:]


def for_prompt(lines: list[str], limit: int = 40) -> str:
    if not lines:
        return ""
    recent = lines[-limit:]
    return "\n".join(f"- {l}" for l in recent)
