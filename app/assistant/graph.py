"""Assistant orchestration graph (LangGraph).

Routing policy (single-entry multi-agent):
    user -> [load_context] -> [classify_intent] -> one of:
        knowledge_qa    -> kb_answer    (Assistant + RAG hybrid retrieval)
        tool_call       -> tool_execute (Assistant -> MCP business tools)
        agent_delegate  -> agent_delegate (Assistant -> A2A specialist)
        chitchat        -> chitchat     (direct LLM)
    -> [persist_memory] -> END

Every node writes an audit record under the same trace_id.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from app.assistant.a2a_client import get_a2a_pool
from app.assistant.intent import IntentRecognizer
from app.assistant.memory import get_memory_store
from app.assistant.mcp_client import get_mcp_pool
from app.assistant.prompts import DIRECT_PROMPT, KB_ANSWER_PROMPT
from app.config import get_settings
from app.rag.retriever import HybridRetriever
from app.schemas import ChatRequest, ChatResponse, IntentResult, IntentType, Role
from app.security.audit import get_audit_logger, new_trace_id
from app.security.auth import (
    PermissionDenied,
    check_agent_permission,
    check_mcp_permission,
    filter_tools_for_role,
)
from app.security.masking import mask_text


class AssistantState(TypedDict):
    """State carried through the orchestration graph."""

    message: str
    session_id: str
    user_id: str
    role: Role
    trace_id: str
    history: str
    intent: IntentResult | None
    answer: str
    route: Literal["assistant_kb", "mcp_tool", "a2a_agent", "direct"]
    target: str | None


class AssistantOrchestrator:
    """Single-entry Assistant that routes across KB / MCP / A2A layers."""

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0.3)
        self._intent = IntentRecognizer()
        self._memory = get_memory_store()
        self._audit = get_audit_logger()
        self._retriever: HybridRetriever | None = None
        self._graph = self._build_graph()

    # ---------------- graph nodes ----------------

    async def load_context(self, state: AssistantState) -> dict[str, Any]:
        history = self._memory.history_text(state["session_id"])
        return {"history": history}

    async def classify_intent(self, state: AssistantState) -> dict[str, Any]:
        intent = await self._intent.classify(state["message"], state.get("history", ""))
        self._audit.log(
            state["trace_id"], "assistant", "intent_classified", intent.model_dump(), state["session_id"]
        )
        return {"intent": intent, "target": intent.target}

    # ---------------- graph nodes ----------------
    async def kb_answer(self, state: AssistantState) -> dict[str, Any]:
        retriever = await self._get_retriever()
        chunks = await retriever.retrieve(state["message"])
        context = retriever.format_context(chunks) if chunks else "(知识库暂无相关资料)"
        prompt = KB_ANSWER_PROMPT.format(
            context=context, history=state.get("history", "(无)"), message=state["message"]
        )
        resp = await self._llm.ainvoke(prompt)
        self._audit.log(
            state["trace_id"], "assistant", "kb_answered",
            {"chunks": [c.chunk_id for c in chunks]}, state["session_id"],
        )
        return {"answer": str(resp.content), "route": "assistant_kb"}

    async def tool_execute(self, state: AssistantState) -> dict[str, Any]:
        """Run a small ReAct loop over the target domain's MCP tools."""
        intent = state["intent"]
        target = intent.target if intent and intent.target else "hr"
        try:
            check_mcp_permission(state["role"], target, "*")
        except PermissionDenied as exc:
            return {"answer": f"权限不足:{exc}", "route": "mcp_tool", "target": target}

        all_tools = await get_mcp_pool().get_tools(target)
        # 权限Mask: 按角色×工具白名单矩阵过滤, 隐藏工具对 LLM 不可见、不可调。
        tools = filter_tools_for_role(state["role"], target, all_tools)
        visible_names = [t.name for t in tools]
        self._audit.log(
            state["trace_id"], "assistant", "tools_filtered",
            {"server": target, "role": state["role"].value, "visible_tools": visible_names},
            state["session_id"],
        )
        if not tools:
            return {
                "answer": f"权限不足: 角色 {state['role'].value} 在 {target} 域无可用工具。",
                "route": "mcp_tool", "target": target,
            }

        agent = create_react_agent(self._llm, tools)
        task = f"[employee_id={state['user_id']}] {state['message']}"
        self._audit.log(state["trace_id"], "assistant", "mcp_dispatch", {"server": target}, state["session_id"])
        result = await agent.ainvoke({"messages": [("user", task)]})
        answer = "工具调用未产生回复。"
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                answer = str(msg.content)
                break
        return {"answer": answer, "route": "mcp_tool", "target": target}

    async def agent_delegate(self, state: AssistantState) -> dict[str, Any]:
        """Delegate to a specialist agent over the A2A protocol."""
        intent = state["intent"]
        target = intent.target if intent and intent.target else "hr"
        agent_name = f"{target}_agent"
        try:
            check_agent_permission(state["role"], agent_name)
        except PermissionDenied as exc:
            return {"answer": f"权限不足:{exc}", "route": "a2a_agent", "target": target}

        # Carry employee identity + recent context so the specialist can act directly.
        task = f"[employee_id={state['user_id']}] {state['message']}"
        if state.get("history"):
            task = f"对话背景:\n{state['history']}\n\n当前请求: {task}"
        self._audit.log(state["trace_id"], "assistant", "a2a_delegate", {"agent": agent_name}, state["session_id"])
        # 可信身份经协议级 metadata 结构化下发 (而非文本标签), 供专业智能体做权限分级。
        answer = await get_a2a_pool().send(
            target,
            task,
            metadata={"user_id": state["user_id"], "role": state["role"].value},
        )
        return {"answer": answer, "route": "a2a_agent", "target": target}

    async def chitchat(self, state: AssistantState) -> dict[str, Any]:
        resp = await self._llm.ainvoke(
            f"{DIRECT_PROMPT}\n\n用户: {state['message']}"
        )
        return {"answer": str(resp.content), "route": "direct"}

    async def persist_memory(self, state: AssistantState) -> dict[str, Any]:
        masked_answer = mask_text(state["answer"])
        await self._memory.append(state["session_id"], mask_text(state["message"]), masked_answer)
        self._audit.log(
            state["trace_id"], "assistant", "turn_completed",
            {"route": state.get("route"), "answer_len": len(state["answer"])}, state["session_id"],
        )
        return {}

    # ---------------- routing ----------------

    @staticmethod
    def _route_by_intent(state: AssistantState) -> str:
        intent = state["intent"]
        if intent is None:
            return "kb_answer"
        return {
            IntentType.KNOWLEDGE_QA: "kb_answer",
            IntentType.TOOL_CALL: "tool_execute",
            IntentType.AGENT_DELEGATE: "agent_delegate",
            IntentType.CHITCHAT: "chitchat",
        }[intent.intent]

    def _build_graph(self):
        g = StateGraph(AssistantState)
        g.add_node("load_context", self.load_context)
        g.add_node("classify_intent", self.classify_intent)
        g.add_node("kb_answer", self.kb_answer)
        g.add_node("tool_execute", self.tool_execute)
        g.add_node("agent_delegate", self.agent_delegate)
        g.add_node("chitchat", self.chitchat)
        g.add_node("persist_memory", self.persist_memory)

        g.add_edge(START, "load_context")
        g.add_edge("load_context", "classify_intent")
        g.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "kb_answer": "kb_answer",
                "tool_execute": "tool_execute",
                "agent_delegate": "agent_delegate",
                "chitchat": "chitchat",
            },
        )
        for node in ("kb_answer", "tool_execute", "agent_delegate", "chitchat"):
            g.add_edge(node, "persist_memory")
        g.add_edge("persist_memory", END)
        return g.compile()

    # ---------------- public API ----------------

    async def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
            self._retriever.rebuild_bm25()
        return self._retriever

    async def handle(self, req: ChatRequest) -> ChatResponse:
        """Handle one user turn end-to-end."""
        trace_id = new_trace_id()
        self._audit.log(
            trace_id, "user", "message_received",
            {"user_id": req.user_id, "role": req.role.value, "message": req.message},
            req.session_id,
        )
        final: AssistantState = await self._graph.ainvoke(
            {
                "message": req.message,
                "session_id": req.session_id,
                "user_id": req.user_id,
                "role": req.role,
                "trace_id": trace_id,
                "history": "",
                "intent": None,
                "answer": "",
                "route": "direct",
                "target": None,
            }
        )
        intent = final.get("intent") or IntentResult(intent=IntentType.CHITCHAT)
        return ChatResponse(
            session_id=req.session_id,
            answer=mask_text(final["answer"]),
            intent=intent.intent,
            route=final.get("route", "direct"),
            target=final.get("target"),
            trace_id=trace_id,
            metadata={"confidence": intent.confidence, "reason": intent.reason},
        )


_orchestrator: AssistantOrchestrator | None = None


def get_orchestrator() -> AssistantOrchestrator:
    """Process-wide singleton orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AssistantOrchestrator()
    return _orchestrator
