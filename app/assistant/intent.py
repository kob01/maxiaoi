"""Intent recognition powered by the small local model (qwen3.5).

Falls back to keyword heuristics when the model output cannot be parsed,
so routing never crashes on bad LLM output.
"""

from __future__ import annotations

import json
import re

from langchain_ollama import ChatOllama

from app.assistant.prompts import INTENT_PROMPT
from app.config import get_settings
from app.schemas import IntentResult, IntentType

_AGENT_KEYWORDS = {
    "finance": ("报销", "费用", "发票", "借款", "付款", "预算"),
    "hr": ("入职", "离职", "在职证明", "证明", "考勤", "请假", "年假申请", "工单"),
}
_TOOL_PATTERNS = re.compile(r"(FIN\d+|HR\d+|余额|进度查询|查询单号)")


class IntentRecognizer:
    """LLM-based intent classifier with deterministic fallback."""

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatOllama(
            model=settings.intent_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            format="json",
        )

    def _fallback(self, message: str) -> IntentResult:
        """Keyword-based routing when the model fails."""
        for target, kws in _AGENT_KEYWORDS.items():
            if any(k in message for k in kws):
                if _TOOL_PATTERNS.search(message):
                    return IntentResult(intent=IntentType.TOOL_CALL, target=target, confidence=0.55, reason="keyword:tool")
                return IntentResult(intent=IntentType.AGENT_DELEGATE, target=target, confidence=0.55, reason="keyword:agent")
        return IntentResult(intent=IntentType.KNOWLEDGE_QA, confidence=0.4, reason="keyword:default_kb")

    async def classify(self, message: str, history: str) -> IntentResult:
        """Classify the latest user utterance."""
        prompt = INTENT_PROMPT.format(history=history or "(无)", message=message)
        try:
            resp = await self._llm.ainvoke(prompt)
            data = json.loads(str(resp.content))
            intent = IntentType(data.get("intent", "knowledge_qa"))
            target = data.get("target")
            if target not in ("finance", "hr"):
                target = None
            return IntentResult(
                intent=intent,
                target=target,
                confidence=float(data.get("confidence", 0.5)),
                reason=str(data.get("reason", "")),
            )
        except Exception:
            return self._fallback(message)
