"""Finance reimbursement MCP Server (FastMCP, streamable-http transport).

Exposes the enterprise finance system as MCP tools. A real deployment would
proxy the actual ERP/expense APIs; here an in-memory ledger keeps the demo
self-contained and deterministic.

Run:
    python -m app.mcp_servers.finance_server     # serves http://0.0.0.0:8002/mcp
"""

from __future__ import annotations

import itertools
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("finance-reimbursement-system", host="0.0.0.0", port=8002)

_order_seq = itertools.count(5000)
_ORDERS: dict[str, dict[str, Any]] = {}

ALLOWED_CATEGORIES = {"差旅费", "交通费", "餐饮费", "办公用品", "培训费"}
SINGLE_LIMIT = 5000.0  # per-order limit (CNY)

# In-memory department budgets for the finance_budget_query tool.
_BUDGETS: dict[str, dict[str, float]] = {
    "研发部": {"annual": 200000.0, "used": 135000.0},
    "市场部": {"annual": 350000.0, "used": 210000.0},
    "人事部": {"annual": 120000.0, "used": 48000.0},
    "财务部": {"annual": 90000.0, "used": 12000.0},
}


@mcp.tool()
def create_reimbursement(user_id: str, title: str, amount: float, category: str, reason: str = "") -> dict[str, Any]:
    """Submit a reimbursement order.

    Args:
        user_id: Employee ID of the claimant.
        title: Expense title, e.g. 上海出差高铁票.
        amount: Amount in CNY; must be positive and <= 5000 per order.
        category: One of 差旅费/交通费/餐饮费/办公用品/培训费.
        reason: Business justification (optional).

    Returns:
        Created order with order_no and workflow status, or error payload.
    """
    if category not in ALLOWED_CATEGORIES:
        return {"error": f"非法报销类别: {category}; 可选: {sorted(ALLOWED_CATEGORIES)}"}
    if amount <= 0:
        return {"error": "金额必须大于 0"}
    if amount > SINGLE_LIMIT:
        return {"error": f"单笔报销上限 {SINGLE_LIMIT} 元, 请拆分后提交"}

    order_no = f"FIN{next(_order_seq)}"
    order = {
        "order_no": order_no,
        "user_id": user_id,
        "title": title,
        "amount": round(amount, 2),
        "category": category,
        "reason": reason,
        "status": "SUBMITTED",
        "current_node": "部门主管审批",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _ORDERS[order_no] = order
    return order


@mcp.tool()
def query_reimbursement(order_no: str) -> dict[str, Any]:
    """Query a reimbursement order by order number.

    Args:
        order_no: Order number, e.g. FIN5000.

    Returns:
        Order record or error payload.
    """
    order = _ORDERS.get(order_no)
    if order is None:
        return {"error": f"order {order_no} not found"}
    return order


@mcp.tool()
def list_reimbursements(user_id: str) -> list[dict[str, Any]]:
    """List all reimbursement orders of an employee.

    Args:
        user_id: Employee ID.

    Returns:
        List of orders (may be empty).
    """
    return [o for o in _ORDERS.values() if o["user_id"] == user_id]


@mcp.tool()
def finance_budget_query(department: str) -> dict[str, Any]:
    """Query a department's annual budget usage.

    注意: 本工具为敏感工具, 仅对部门经理/HR/财务专员等管理角色可见。

    Args:
        department: Department name, e.g. 研发部.

    Returns:
        Budget summary with annual / used / remaining, or error payload.
    """
    budget = _BUDGETS.get(department)
    if budget is None:
        return {"error": f"未知部门: {department}; 可选: {sorted(_BUDGETS)}"}
    remaining = round(budget["annual"] - budget["used"], 2)
    return {
        "department": department,
        "annual": budget["annual"],
        "used": budget["used"],
        "remaining": remaining,
    }


@mcp.tool()
def get_reimbursement_policy(category: str) -> dict[str, Any]:
    """Fetch the reimbursement policy snippet for a category.

    Args:
        category: Expense category.

    Returns:
        Policy description with limit and required attachments.
    """
    policies = {
        "差旅费": "差旅费按城市等级限额, 需附行程单/发票, 单笔≤5000元",
        "交通费": "市内交通实报实销, 需附发票, 单笔≤5000元",
        "餐饮费": "业务招待需事前审批, 需附发票与接待清单",
        "办公用品": "需附采购清单与发票, 单笔≤5000元",
        "培训费": "需培训通知与发票, 年度额度20000元",
    }
    if category not in policies:
        return {"error": f"未知类别: {category}; 可选: {sorted(policies)}"}
    return {"category": category, "policy": policies[category]}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
