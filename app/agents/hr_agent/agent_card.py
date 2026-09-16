"""HR_Agent A2A Agent Card definition."""

from __future__ import annotations

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from app.config import get_settings


def build_agent_card() -> AgentCard:
    """Build the HR_Agent card describing its A2A endpoint & skills."""
    settings = get_settings()
    return AgentCard(
        name="HR_Agent",
        description=(
            "HR 专业智能体:负责 HR 工单全流程(工单创建/查询/取消)与年假余额查询,"
            "通过 MCP 协议调用 HR 工单系统完成实际操作。"
        ),
        url=f"{settings.hr_agent_url}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
        skills=[
            AgentSkill(
                id="create_hr_ticket",
                name="HR 工单创建",
                description="按类别(入职/离职/考勤/薪酬/证明开具/其他)创建 HR 服务工单",
                tags=["hr", "ticket"],
                examples=["帮我开一个在职证明", "我要提一个考勤异常工单"],
            ),
            AgentSkill(
                id="query_hr_ticket",
                name="HR 工单查询",
                description="按工单号查询状态,或列出本人全部工单",
                tags=["hr", "query"],
                examples=["HR1000 处理到哪了"],
            ),
            AgentSkill(
                id="leave_balance",
                name="年假余额查询",
                description="查询员工年假总额/已用/剩余",
                tags=["hr", "leave"],
                examples=["我还有几天年假"],
            ),
        ],
    )
