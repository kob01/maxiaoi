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

import logging
from typing import Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from app.assistant.a2a_client import get_a2a_pool
from app.assistant.intent import IntentRecognizer
from app.assistant.mcp_client import get_mcp_pool
from app.assistant.memory import get_memory_store
from app.assistant.prompts import DIRECT_PROMPT, KB_ANSWER_PROMPT
from app.config import get_settings
from app.llm import get_chat_model
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

logger = logging.getLogger(__name__)


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
    docs_meta: list[dict[str, Any]]


class AssistantOrchestrator:
    """Single-entry Assistant that routes across KB / MCP / A2A layers."""

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._llm = get_chat_model(settings.llm_model, temperature=0.3)
        self._intent = IntentRecognizer()
        self._memory = get_memory_store()
        self._audit = get_audit_logger()
        self._retriever: HybridRetriever | None = None
        self._graph = self._build_graph()

    # ---------------- graph nodes ----------------

    async def load_context(self, state: AssistantState) -> dict[str, Any]:
        history = self._memory.history_text(state.get("session_id") or "")
        return {"history": history}

    async def classify_intent(self, state: AssistantState) -> dict[str, Any]:
        intent = await self._intent.classify(state["message"], state.get("history", ""))
        self._audit.log(
            state.get("trace_id") or new_trace_id(), "assistant", "intent_classified",
            intent.model_dump(), state.get("session_id"),
        )
        return {"intent": intent, "target": intent.target}

    # ---------------- graph nodes ----------------
    async def kb_answer(self, state: AssistantState) -> dict[str, Any]:
        retriever = await self._get_retriever()
        children = await retriever.retrieve(state["message"])
        # Assemble hit child chunks into complete parent section blocks so
        # the LLM answers from full sections (with page/section citations).
        chunks = retriever.assemble_parents(children) if children else []
        meta_map: dict[str, dict[str, Any]] = {}
        if chunks:
            try:
                from app.docs.service import get_meta_map

                meta_map = await get_meta_map(list({c.doc_id for c in chunks}))
            except Exception as exc:  # MySQL down must not break chat
                logger.warning("doc metadata lookup failed, degrade to plain context: %s", exc)
        context = retriever.format_context(chunks, meta_map) if chunks else "(知识库暂无相关资料)"
        prompt = KB_ANSWER_PROMPT.format(
            context=context, history=state.get("history", "(无)"), message=state["message"]
        )
        resp = await self._llm.ainvoke(prompt)
        self._audit.log(
            state.get("trace_id") or "", "assistant", "kb_answered",
            {"chunks": [c.chunk_id for c in chunks]}, state.get("session_id"),
        )
        docs_meta = [
            {
                "doc_key": c.doc_id,
                "title": c.title,
                "section": c.section,
                "page_no": c.page_no,
                "tags": (meta_map.get(c.doc_id) or {}).get("tags", []),
            }
            for c in chunks
        ]
        return {"answer": str(resp.content), "route": "assistant_kb", "docs_meta": docs_meta}

    async def tool_execute(self, state: AssistantState) -> dict[str, Any]:
        """Run a small ReAct loop over the target domain's MCP tools."""
        intent = state["intent"]
        target = intent.target if intent and intent.target else "hr"
        role = self._role_of(state)
        try:
            check_mcp_permission(role, target, "*")
        except PermissionDenied as exc:
            return {"answer": f"权限不足:{exc}", "route": "mcp_tool", "target": target}

        all_tools = await get_mcp_pool().get_tools(target)
        # 权限Mask: 按角色×工具白名单矩阵过滤, 隐藏工具对 LLM 不可见、不可调。
        tools = filter_tools_for_role(role, target, all_tools)
        visible_names = [t.name for t in tools]
        self._audit.log(
            state.get("trace_id") or "", "assistant", "tools_filtered",
            {"server": target, "role": role.value, "visible_tools": visible_names},
            state.get("session_id"),
        )
        if not tools:
            return {
                "answer": f"权限不足: 角色 {role.value} 在 {target} 域无可用工具。",
                "route": "mcp_tool", "target": target,
            }

        agent = create_agent(self._llm, tools)
        task = f"[employee_id={state.get('user_id') or 'anonymous'}] {state['message']}"
        self._audit.log(state.get("trace_id") or "", "assistant", "mcp_dispatch", {"server": target}, state.get("session_id"))
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
        role = self._role_of(state)
        try:
            check_agent_permission(role, agent_name)
        except PermissionDenied as exc:
            return {"answer": f"权限不足:{exc}", "route": "a2a_agent", "target": target}

        # Carry employee identity + recent context so the specialist can act directly.
        task = f"[employee_id={state.get('user_id') or 'anonymous'}] {state['message']}"
        if state.get("history"):
            task = f"对话背景:\n{state['history']}\n\n当前请求: {task}"
        self._audit.log(state.get("trace_id") or "", "assistant", "a2a_delegate", {"agent": agent_name}, state.get("session_id"))
        # 可信身份经协议级 metadata 结构化下发 (而非文本标签), 供专业智能体做权限分级。
        answer = await get_a2a_pool().send(
            target,
            task,
            metadata={"user_id": state.get("user_id") or "anonymous", "role": role.value},
        )
        return {"answer": answer, "route": "a2a_agent", "target": target}

    async def chitchat(self, state: AssistantState) -> dict[str, Any]:
        resp = await self._llm.ainvoke(
            f"{DIRECT_PROMPT}\n\n用户: {state['message']}"
        )
        return {"answer": str(resp.content), "route": "direct"}

    async def persist_memory(self, state: AssistantState) -> dict[str, Any]:
        masked_answer = mask_text(state["answer"])
        await self._memory.append(state.get("session_id") or "", mask_text(state["message"]), masked_answer)
        self._audit.log(
            state.get("trace_id") or "", "assistant", "turn_completed",
            {"route": state.get("route"), "answer_len": len(state["answer"])}, state.get("session_id"),
        )
        return {}

    # ---------------- routing ----------------

    @staticmethod
    def _role_of(state: AssistantState) -> Role:
        """Normalise role; LangGraph Studio may pass a plain string."""
        role = state.get("role")
        if isinstance(role, Role):
            return role
        try:
            return Role(str(role))
        except ValueError:
            return Role.EMPLOYEE

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

    async def refresh_knowledge(self) -> None:
        """Rebuild the BM25 channel after a document (re)ingest."""
        if self._retriever is not None:
            self._retriever.rebuild_bm25()

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
                "docs_meta": [],
            }
        )
        intent = final.get("intent") or IntentResult(intent=IntentType.CHITCHAT)
        self._audit.log(
            trace_id,
            "handle",
            "final",
            {
                "intent": intent.intent,
                "confidence": intent.confidence,
                "reason": intent.reason,
                "answer": final["answer"],
                "route": final.get("route", "direct"),
                "target": final.get("target"),
                "_memory": self._memory.history_text(req.session_id),
            },
            req.session_id,
        )
        return ChatResponse(
            session_id=req.session_id,
            answer=mask_text(final["answer"]),
            intent=intent.intent,
            route=final.get("route", "direct"),
            target=final.get("target"),
            trace_id=trace_id,
            metadata={
                "confidence": intent.confidence,
                "reason": intent.reason,
                "docs": final.get("docs_meta", []),
            },
        )


_orchestrator: AssistantOrchestrator | None = None


def get_orchestrator() -> AssistantOrchestrator:
    """Process-wide singleton orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AssistantOrchestrator()
    return _orchestrator


def get_graph():
    """Graph factory exposed to LangGraph Studio (see langgraph.json).

    Enables LangSmith tracing first so every Studio run is captured as a
    trace under the configured project.
    """
    from app.tracing import init_tracing

    init_tracing()
    return get_orchestrator()._graph
