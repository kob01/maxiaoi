"""Chat model factory.

Routes to the DeepSeek online API (OpenAI-compatible) for model names
starting with ``deepseek``, and to local Ollama otherwise. Callers only ask
for a chat model by name/temperature and never care about the transport.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import get_settings


def is_deepseek(model: str) -> bool:
    """Whether the given model name should be served by the DeepSeek API."""
    return model.lower().startswith("deepseek")


def get_chat_model(
    model: str | None = None,
    *,
    temperature: float = 0.1,
    json_mode: bool = False,
) -> BaseChatModel:
    """Build a chat model for ``model`` (defaults to ``settings.llm_model``).

    ``json_mode`` requests JSON-structured output (intent classification).
    """
    settings = get_settings()
    name = model or settings.llm_model

    if is_deepseek(name):
        from langchain_openai import ChatOpenAI

        if not settings.deepseek_api_key:
            raise RuntimeError(
                "缺少 DeepSeek API 密钥: 本地开发请在 .env 设置 DEEPSEEK_API_KEY, "
                "Docker 部署请创建 docker/secrets/deepseek_api_key.txt "
                "(容器内挂载为 /run/secrets/deepseek_api_key)"
            )
        kwargs: dict = {
            "model": name,
            "base_url": settings.deepseek_base_url,
            "api_key": settings.deepseek_api_key,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
            # 意图分类等结构化短任务关闭思考模式, 降低时延 (deepseek-flash 默认开启)
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        return ChatOpenAI(**kwargs)

    from langchain_ollama import ChatOllama

    ollama_kwargs: dict = {
        "model": name,
        "base_url": settings.ollama_base_url,
        "temperature": temperature,
    }
    if json_mode:
        ollama_kwargs["format"] = "json"
    return ChatOllama(**ollama_kwargs)
