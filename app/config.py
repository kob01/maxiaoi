"""Application settings loaded from environment / .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 敏感项对应的 Docker secret 文件(BuildKit/compose secrets 挂载路径)
_SECRET_FILES = {
    "mysql_password": "/run/secrets/mysql_password",
    "deepseek_api_key": "/run/secrets/deepseek_api_key",
}


class Settings(BaseSettings):
    """Central configuration for the whole platform.

    Real environment variables take precedence over the .env files
    (.env and docker/.env are both supported for local convenience).
    """

    model_config = SettingsConfigDict(
        env_file=(".env", "docker/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        # 字段取默认值(环境变量缺失)时也运行校验器, 以便回退读取 /run/secrets/*
        validate_default=True,
    )

    # Ollama (仅 embedding / rerank / vision 仍走本地 Ollama)
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    rerank_model: str = "dengcao/bge-reranker-v2-m3"

    # DeepSeek 在线 API (OpenAI 兼容格式, 用于 LLM / 意图识别)
    deepseek_base_url: str = "https://api.deepseek.com"
    # 密钥优先取环境变量, 其次由校验器读取 /run/secrets/deepseek_api_key
    deepseek_api_key: str = ""
    llm_model: str = "deepseek-flash"
    intent_model: str = "deepseek-flash"

    # RAG (env var named MILVUS_LITE_URI to avoid clashing with pymilvus's own MILVUS_URI)
    milvus_lite_uri: str = "./data/milvus_lite.db"
    milvus_collection: str = "enterprise_knowledge"
    knowledge_dir: str = "./data/knowledge"
    rag_top_k: int = 8
    rerank_top_n: int = 4
    # Rerank fails -> graceful fallback to RRF fusion order (e.g. Windows
    # Ollama llama.cpp crashes on bge-reranker GGUF); set false to skip.
    rerank_enabled: bool = True

    # Document upload & metadata (MySQL)
    mysql_host: str = "47.116.208.170"
    mysql_port: int = 3306
    mysql_user: str = "sql47_116_208_1"
    mysql_password: str = ""
    mysql_database: str = "sql47_116_208_1"
    mysql_connect_timeout: int = 10
    upload_dir: str = "./data/uploads"
    upload_max_mb: int = 50
    # Vision model used to caption uploaded images (needs `ollama pull qwen3-vl`)
    vision_model: str = "qwen3-vl"
    vision_timeout: int = 300
    # Parent-child chunking: a section block larger than this is window-split
    # into child chunks; smaller blocks stay a single child under the parent.
    parent_chunk_max: int = 1200

    # Assistant service
    assistant_host: str = "0.0.0.0"
    assistant_port: int = 8000
    memory_max_turns: int = 10
    memory_summary_threshold: int = 20

    # MCP servers
    hr_mcp_url: str = "http://localhost:8001/mcp"
    finance_mcp_url: str = "http://localhost:8002/mcp"

    # A2A agents
    hr_agent_url: str = "http://localhost:9001"
    finance_agent_url: str = "http://localhost:9002"

    # Security
    audit_log_path: str = "./logs/audit.jsonl"

    @field_validator("mysql_password", "deepseek_api_key", mode="after")
    @classmethod
    def _read_from_secret_file(cls, value: str, info) -> str:
        """环境变量/.env 未提供时, 回退读取 compose secret 文件。"""
        if value:
            return value
        secret_path = Path(_SECRET_FILES[info.field_name])
        if secret_path.is_file():
            return secret_path.read_text(encoding="utf-8").strip()
        return value

    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
