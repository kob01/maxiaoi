"""MySQL engine/session factory.

Password policy: the MySQL password is injected via the MYSQL_PASSWORD
environment variable (or the .env file picked up by pydantic-settings).
It is never hard-coded and .env files are git-ignored.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.config import get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None


def _ensure_password() -> str:
    """Read the MySQL password from the environment (MYSQL_PASSWORD)."""
    password = get_settings().mysql_password
    if not password:
        raise RuntimeError(
            "缺少 MySQL 密码: 请设置环境变量 MYSQL_PASSWORD, "
            "或在项目根目录 .env / docker/.env 中添加 MYSQL_PASSWORD=..., "
            "Docker 部署则创建 docker/secrets/mysql_password.txt"
            "(容器内挂载为 /run/secrets/mysql_password)"
        )
    return password


def get_engine() -> AsyncEngine:
    """Lazily create the async engine (reads MYSQL_PASSWORD on first use)."""
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        password = _ensure_password()
        url = (
            f"mysql+aiomysql://{settings.mysql_user}:{password}"
            f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
            f"?charset=utf8mb4"
        )
        _engine = create_async_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"connect_timeout": settings.mysql_connect_timeout},
        )
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_session_factory() -> async_sessionmaker:
    """Return the session factory (creates the engine on first call)."""
    get_engine()
    assert _session_factory is not None
    return _session_factory


async def init_schema() -> None:
    """Create all metadata tables if they do not exist yet."""
    from app.db.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def db_available() -> bool:
    """Whether the engine has been initialised (password already provided)."""
    return _engine is not None
