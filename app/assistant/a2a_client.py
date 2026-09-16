"""A2A client used by the Assistant to delegate tasks to specialist agents.

Implements the Agent2Agent protocol client flow:
1. Discover the agent card from /.well-known/agent-card.json
2. Send a JSON-RPC 2.0 `message/send` request
3. Unwrap the returned Task/Message into plain text
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TextPart,
)

from app.config import get_settings

AGENT_URLS = {
    "finance": lambda: get_settings().finance_agent_url,
    "hr": lambda: get_settings().hr_agent_url,
}


class A2AClientPool:
    """Lazily-resolved A2A clients keyed by agent domain."""

    def __init__(self) -> None:
        self._clients: dict[str, A2AClient] = {}
        self._http: httpx.AsyncClient | None = None

    async def _get_client(self, domain: str) -> A2AClient:
        """Resolve agent card and build an A2AClient for a domain."""
        if domain in self._clients:
            return self._clients[domain]
        base_url = AGENT_URLS[domain]()
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        card = await A2ACardResolver(httpx_client=self._http, base_url=base_url).get_agent_card()
        client = A2AClient(httpx_client=self._http, agent_card=card)
        self._clients[domain] = client
        return client

    @staticmethod
    def _extract_text(result: Any) -> str:
        """Pull concatenated text parts out of a Task or Message result."""
        parts = []
        if isinstance(result, Task):
            # Prefer the agent's final status message, then artifacts.
            if result.status.message and result.status.message.parts:
                parts = [p.root.text for p in result.status.message.parts if isinstance(p.root, TextPart)]
            if not parts and result.artifacts:
                for artifact in result.artifacts:
                    parts.extend(p.root.text for p in artifact.parts if isinstance(p.root, TextPart))
        elif isinstance(result, Message):
            parts = [p.root.text for p in result.parts if isinstance(p.root, TextPart)]
        return "\n".join(parts) or "(专业智能体未返回文本)"

    async def send(
        self,
        domain: str,
        text: str,
        context_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Delegate a task to a specialist agent and return its reply text.

        Args:
            domain: 'finance' or 'hr'.
            text: Task description in natural language.
            context_id: Optional A2A context id to continue a remote task.
            metadata: Optional structured message metadata (protocol-native),
                e.g. trusted identity {"user_id": ..., "role": ...}.

        Returns:
            The agent's textual answer.
        """
        client = await self._get_client(domain)
        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            messageId=uuid.uuid4().hex,
            contextId=context_id,
            metadata=metadata,
        )
        request = SendMessageRequest(id=uuid.uuid4().hex, params=MessageSendParams(message=message))
        response = await client.send_message(request)
        root = response.root  # JSONRPCErrorResponse | SendMessageSuccessResponse
        error = getattr(root, "error", None)
        if error is not None:
            return f"A2A 调用失败: {getattr(error, 'message', error)}"
        return self._extract_text(getattr(root, "result", None))


_a2a_pool: A2AClientPool | None = None


def get_a2a_pool() -> A2AClientPool:
    """Process-wide singleton A2A client pool."""
    global _a2a_pool
    if _a2a_pool is None:
        _a2a_pool = A2AClientPool()
    return _a2a_pool
