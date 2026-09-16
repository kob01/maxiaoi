"""Shared Pydantic schemas for API layer and internal routing."""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Enterprise roles used for permission whitelists."""

    EMPLOYEE = "employee"
    MANAGER = "manager"
    HR = "hr"
    FINANCE = "finance"
    ADMIN = "admin"


class IntentType(str, Enum):
    """Top-level intent categories produced by the intent recognizer."""

    KNOWLEDGE_QA = "knowledge_qa"      # simple query -> RAG answer
    TOOL_CALL = "tool_call"            # complex operation -> MCP tool
    AGENT_DELEGATE = "agent_delegate"  # professional task -> A2A agent
    CHITCHAT = "chitchat"              # small talk -> direct LLM answer


class IntentResult(BaseModel):
    """Structured output of the intent recognizer."""

    intent: IntentType
    target: Optional[str] = Field(
        default=None,
        description="Sub-target, e.g. 'finance' / 'hr' for agents, or a tool domain.",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="", description="Why the classifier picked this intent.")


class ChatRequest(BaseModel):
    """Inbound chat request from the Web UI / API clients."""

    session_id: str = Field(description="Conversation session identifier.")
    user_id: str = Field(description="End-user identifier.")
    role: Role = Field(default=Role.EMPLOYEE, description="Caller role for permission checks.")
    message: str = Field(description="User utterance.")


class ChatResponse(BaseModel):
    """Outbound chat response."""

    session_id: str
    answer: str
    intent: IntentType
    route: Literal["assistant_kb", "mcp_tool", "a2a_agent", "direct"]
    target: Optional[str] = None
    trace_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeChunk(BaseModel):
    """A single retrievable knowledge chunk."""

    chunk_id: str
    doc_id: str
    title: str
    content: str
    source: str
    modality: Literal["text", "video_transcript"] = "text"
    score: float = 0.0
