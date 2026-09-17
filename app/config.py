"""Application settings loaded from environment / .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the whole platform.

    Real environment variables take precedence over the .env files
    (.env and docker/.env are both supported for local convenience).
    """

    model_config = SettingsConfigDict(
        env_file=(".env", "docker/.env"), env_file_encoding="utf-8", extra="ignore"
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3.5"
    intent_model: str = "qwen3.5"
    embedding_model: str = "bge-m3"
    rerank_model: str = "dengcao/bge-reranker-v2-m3"

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
    # 密码经环境变量 MYSQL_PASSWORD 注入(优先真实环境变量, 其次 .env 文件);
    # pydantic-settings 统一解析, 不落代码库(.env 已被 .gitignore 排除)。
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

    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
