"""HR_Agent business logic + A2A AgentExecutor."""

from __future__ import annotations

from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import get_settings
from app.llm import get_chat_model
from app.security.audit import get_audit_logger

SYSTEM_PROMPT = """你是 HR_Agent,企业 HR 服务专业智能体。
职责:
1. HR 工单创建:收集类别、标题、描述后调用 create_hr_ticket。
2. 工单查询/取消:调用 query_hr_ticket / list_hr_tickets / cancel_hr_ticket。
3. 年假查询:调用 get_leave_balance。

规则:
- 缺少必填信息时主动追问,不要编造。
- 工具返回的 error 字段必须如实转达。
- 用户消息中可能携带 [employee_id=XXX],视作 HR 工号。
- 用简洁中文回复。"""


class HRAgent:
    """LangGraph ReAct agent over HR MCP tools."""

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._llm = get_chat_model(settings.llm_model, temperature=0)
        self._agent: Any | None = None

    async def _ensure_agent(self) -> Any:
        if self._agent is None:
            client = MultiServerMCPClient(
                {"hr": {"url": self._settings.hr_mcp_url, "transport": "streamable_http"}}
            )
            tools = await client.get_tools()
            self._agent = create_agent(self._llm, tools, prompt=SYSTEM_PROMPT)
        return self._agent

    async def invoke(self, user_text: str) -> str:
        agent = await self._ensure_agent()
        result = await agent.ainvoke({"messages": [("user", user_text)]})
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)
        return "HR 智能体未能生成有效回复。"


class HRAgentExecutor(AgentExecutor):
    """A2A AgentExecutor bridge for HR_Agent."""

    def __init__(self) -> None:
        self._agent = HRAgent()
        self._audit = get_audit_logger()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        trace_id = context.task_id or context.context_id or "unknown"
        self._audit.log(trace_id, "hr_agent", "a2a_task_received", {"input": user_input})
        try:
            answer = await self._agent.invoke(user_input)
        except Exception as exc:
            answer = f"HR 智能体处理失败: {exc}"
        self._audit.log(trace_id, "hr_agent", "a2a_task_completed", {"answer": answer})
        await event_queue.enqueue_event(new_agent_text_message(answer))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("HR_Agent does not support cancellation")
