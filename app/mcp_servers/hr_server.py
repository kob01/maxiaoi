"""HR ticket-system MCP Server (FastMCP, streamable-http transport).

Wraps the enterprise HR backend (mocked here with an in-memory store) as
standard MCP tools so any MCP-compatible client can call them.

Run:
    python -m app.mcp_servers.hr_server          # serves http://0.0.0.0:8001/mcp
"""

from __future__ import annotations

import itertools
import uuid
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hr-ticket-system", host="0.0.0.0", port=8001)

# ---------------- mock backend ----------------
_ticket_seq = itertools.count(1000)
_TICKETS: dict[str, dict[str, Any]] = {}


@mcp.tool()
def create_hr_ticket(user_id: str, category: str, title: str, description: str) -> dict[str, Any]:
    """Create an HR service ticket.

    Args:
        user_id: Employee ID of the requester.
        category: One of 入职/离职/考勤/薪酬/证明开具/其他.
        title: Short ticket title.
        description: Detailed request description.

    Returns:
        The created ticket record including ticket_no and status.
    """
    ticket_no = f"HR{next(_ticket_seq)}"
    ticket = {
        "ticket_no": ticket_no,
        "user_id": user_id,
        "category": category,
        "title": title,
        "description": description,
        "status": "OPEN",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _TICKETS[ticket_no] = ticket
    return ticket


@mcp.tool()
def query_hr_ticket(ticket_no: str) -> dict[str, Any]:
    """Query an HR ticket by its ticket number.

    Args:
        ticket_no: Ticket number, e.g. HR1000.

    Returns:
        Ticket record, or an error payload if not found.
    """
    ticket = _TICKETS.get(ticket_no)
    if ticket is None:
        return {"error": f"ticket {ticket_no} not found"}
    return ticket


@mcp.tool()
def list_hr_tickets(user_id: str) -> list[dict[str, Any]]:
    """List all HR tickets submitted by a given employee.

    Args:
        user_id: Employee ID.

    Returns:
        List of ticket records (may be empty).
    """
    return [t for t in _TICKETS.values() if t["user_id"] == user_id]


@mcp.tool()
def cancel_hr_ticket(ticket_no: str) -> dict[str, Any]:
    """Cancel an OPEN HR ticket.

    Args:
        ticket_no: Ticket number to cancel.

    Returns:
        Updated ticket record or error payload.
    """
    ticket = _TICKETS.get(ticket_no)
    if ticket is None:
        return {"error": f"ticket {ticket_no} not found"}
    if ticket["status"] != "OPEN":
        return {"error": f"ticket {ticket_no} is {ticket['status']}, cannot cancel"}
    ticket["status"] = "CANCELLED"
    return ticket


@mcp.tool()
def get_leave_balance(user_id: str) -> dict[str, Any]:
    """Get annual-leave balance of an employee (mocked deterministic value).

    Args:
        user_id: Employee ID.

    Returns:
        Balance info: total / used / remaining days.
    """
    seed = int(uuid.uuid5(uuid.NAMESPACE_DNS, user_id).hex[:4], 16)
    total = 10 + seed % 6
    used = seed % 5
    return {"user_id": user_id, "annual_leave_total": total, "used": used, "remaining": total - used}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
