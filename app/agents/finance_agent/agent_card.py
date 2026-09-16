"""Finance_Agent A2A Agent Card definition.

The card is served at /.well-known/agent-card.json by the A2A server and
lets clients discover capabilities/skills before delegation, per the
Agent2Agent protocol specification.
"""

from __future__ import annotations

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from app.config import get_settings


def build_agent_card() -> AgentCard:
    """Build the Finance_Agent card describing its A2A endpoint & skills."""
    settings = get_settings()
    return AgentCard(
        name="Finance_Agent",
        description=(
            "财务专业智能体:负责费用报销全流程(政策咨询、报销单创建、进度查询),"
            "通过 MCP 协议调用财务报销系统完成实际操作。"
        ),
        url=f"{settings.finance_agent_url}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
        skills=[
            AgentSkill(
                id="create_reimbursement",
                name="费用报销办理",
                description="根据用户提供的费用信息创建报销单并进入审批流",
                tags=["finance", "reimbursement"],
                examples=["我要报销", "帮我报销一张500元的差旅费发票"],
            ),
            AgentSkill(
                id="query_reimbursement",
                name="报销进度查询",
                description="按单号查询报销单状态与当前审批节点",
                tags=["finance", "query"],
                examples=["FIN5000 到哪一步了"],
            ),
            AgentSkill(
                id="query_budget",
                name="部门预算查询",
                description="查询部门年度预算总额、已用与剩余(仅部门经理/HR/财务专员等管理角色可用)",
                tags=["finance", "budget"],
                examples=["查一下研发部的预算", "市场部预算还剩多少"],
            ),
            AgentSkill(
                id="reimbursement_policy",
                name="报销政策咨询",
                description="解答各类费用报销的额度、附件要求等政策问题",
                tags=["finance", "policy"],
                examples=["餐饮费报销需要什么材料"],
            ),
        ],
    )
