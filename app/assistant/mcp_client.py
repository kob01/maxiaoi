"""MCP client: discovers tools from all configured MCP servers.

Uses langchain-mcp-adapters' MultiServerMCPClient over streamable-http.
Tools are cached after first discovery; call refresh() to re-discover.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import get_settings


class MCPClientPool:
    """Pool of MCP tool connections keyed by server name."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = MultiServerMCPClient(
            {
                "hr": {"url": settings.hr_mcp_url, "transport": "streamable_http"},
                "finance": {"url": settings.finance_mcp_url, "transport": "streamable_http"},
            }
        )
        self._tools: list[BaseTool] | None = None

    async def get_tools(self, server: str | None = None) -> list[BaseTool]:
        """Return LangChain tools, optionally filtered by server name prefix."""
        if self._tools is None:
            self._tools = await self._client.get_tools()
        if server is None:
            return list(self._tools)
        # langchain-mcp-adapters prefixes tool names with the server key is
        # not guaranteed; filter by checking the originating server via the
        # client instead.
        return await self._client.get_tools(server_name=server)

    async def refresh(self) -> None:
        """Drop cached tools so next get_tools() re-discovers."""
        self._tools = None


_pool: MCPClientPool | None = None


def get_mcp_pool() -> MCPClientPool:
    """Process-wide singleton MCP client pool."""
    global _pool
    if _pool is None:
        _pool = MCPClientPool()
    return _pool


async def call_mcp_tool(server: str, tool_name: str, args: dict[str, Any]) -> Any:
    """Directly invoke a single MCP tool (used by the tool_call route)."""
    pool = get_mcp_pool()
    tools = await pool.get_tools(server)
    for tool in tools:
        if tool.name == tool_name:
            return await tool.ainvoke(args)
    raise LookupError(f"MCP tool {server}.{tool_name} not found")
