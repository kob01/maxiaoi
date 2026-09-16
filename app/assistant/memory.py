"""Conversation memory: short-term window + LLM-summarized long-term memory.

In-memory per-session store keeps the demo dependency-free; swap the dict
for Redis in production without touching callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_ollama import ChatOllama

from app.assistant.prompts import SUMMARY_PROMPT
from app.config import get_settings


@dataclass
class SessionMemory:
    """Per-session state: rolling window + compressed long-term summary."""

    turns: list[tuple[str, str]] = field(default_factory=list)  # (user, assistant)
    summary: str = ""


class MemoryStore:
    """Session memory manager with summarization on overflow."""

    def __init__(self) -> None:
        settings = get_settings()
        self._max_turns = settings.memory_max_turns
        self._summary_threshold = settings.memory_summary_threshold
        self._sessions: dict[str, SessionMemory] = {}
        self._summarizer = ChatOllama(
            model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0
        )

    def get(self, session_id: str) -> SessionMemory:
        """Get or create the memory for a session."""
        return self._sessions.setdefault(session_id, SessionMemory())

    def history_text(self, session_id: str) -> str:
        """Render summary + recent window as prompt-ready text."""
        mem = self.get(session_id)
        lines: list[str] = []
        if mem.summary:
            lines.append(f"[历史摘要] {mem.summary}")
        for user, assistant in mem.turns[-self._max_turns :]:
            lines.append(f"用户: {user}")
            lines.append(f"助手: {assistant}")
        return "\n".join(lines)

    async def append(self, session_id: str, user: str, assistant: str) -> None:
        """Append one turn; summarize into long-term memory on overflow."""
        mem = self.get(session_id)
        mem.turns.append((user, assistant))
        if len(mem.turns) > self._summary_threshold:
            overflow = mem.turns[: -self._max_turns]
            mem.turns = mem.turns[-self._max_turns :]
            old = "\n".join(f"用户: {u}\n助手: {a}" for u, a in overflow)
            prompt = SUMMARY_PROMPT.format(history=f"{mem.summary}\n{old}".strip())
            resp = await self._summarizer.ainvoke(prompt)
            mem.summary = str(resp.content).strip()


_memory_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    """Process-wide singleton memory store."""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store
