"""Role-based permission whitelists for tools and agents."""

from __future__ import annotations

from typing import Any

from app.schemas import Role

# Which MCP tool namespaces each role may invoke.
MCP_WHITELIST: dict[Role, set[str]] = {
    Role.EMPLOYEE: {"hr", "finance"},
    Role.MANAGER: {"hr", "finance"},
    Role.HR: {"hr", "finance"},
    Role.FINANCE: {"hr", "finance"},
    Role.ADMIN: {"hr", "finance"},
}

# Which A2A agents each role may delegate to.
AGENT_WHITELIST: dict[Role, set[str]] = {
    Role.EMPLOYEE: {"finance_agent", "hr_agent"},
    Role.MANAGER: {"finance_agent", "hr_agent"},
    Role.HR: {"hr_agent", "finance_agent"},
    Role.FINANCE: {"finance_agent", "hr_agent"},
    Role.ADMIN: {"finance_agent", "hr_agent"},
}

# Fine-grained tool-level restrictions: sensitive tools are limited by role.
TOOL_ROLE_RESTRICTIONS: dict[str, set[Role]] = {
    # e.g. only finance staff may look up *other* people's orders in a
    # real system; here the tools are self-service so all roles pass.
}

# ---------------------------------------------------------------------------
# 角色×工具白名单矩阵 (单一事实来源): 默认拒绝语义 —— 工具未列入某角色的白名单
# 即对该角色完全隐藏 (LLM 看不见、调不到), 而非调用时报错。
# 注意: MCP server 新增工具时必须同步维护本矩阵。
# None 表示该域全量可见。
FINANCE_TOOL_WHITELIST: dict[Role, set[str] | None] = {
    Role.EMPLOYEE: {
        "create_reimbursement",
        "query_reimbursement",
        "list_reimbursements",
        "get_reimbursement_policy",
    },
    Role.MANAGER: {
        "create_reimbursement",
        "query_reimbursement",
        "list_reimbursements",
        "get_reimbursement_policy",
        "finance_budget_query",
    },
    Role.HR: None,      # HR/财务专员/管理员: 财务域全量可见
    Role.FINANCE: None,
    Role.ADMIN: None,
}


def filter_tools_for_role(role: Role, server_name: str, tools: list[Any]) -> list[Any]:
    """Return the subset of ``tools`` visible to ``role`` on ``server_name``.

    Tools must expose a ``name`` attribute (LangChain BaseTool). Only the
    finance domain is tiered for now; other domains pass through unchanged.
    Default-deny: roles missing from the matrix see nothing.
    """
    if server_name != "finance":
        return list(tools)
    if role not in FINANCE_TOOL_WHITELIST:
        return []
    whitelist = FINANCE_TOOL_WHITELIST[role]
    if whitelist is None:
        return list(tools)
    return [t for t in tools if t.name in whitelist]


class PermissionDenied(Exception):
    """Raised when a role attempts to use a forbidden tool/agent."""


def check_agent_permission(role: Role, agent_name: str) -> None:
    """Raise PermissionDenied if the role may not delegate to the agent."""
    allowed = AGENT_WHITELIST.get(role, set())
    if agent_name not in allowed:
        raise PermissionDenied(f"角色 {role.value} 无权访问智能体 {agent_name}")


def check_mcp_permission(role: Role, server_name: str, tool_name: str) -> None:
    """Raise PermissionDenied if the role may not call the MCP tool."""
    allowed_servers = MCP_WHITELIST.get(role, set())
    if server_name not in allowed_servers:
        raise PermissionDenied(f"角色 {role.value} 无权访问 MCP 服务 {server_name}")
    restricted = TOOL_ROLE_RESTRICTIONS.get(f"{server_name}.{tool_name}")
    if restricted is not None and role not in restricted:
        raise PermissionDenied(f"角色 {role.value} 无权调用工具 {server_name}.{tool_name}")
