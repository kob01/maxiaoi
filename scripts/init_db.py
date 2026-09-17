"""MySQL metadata bootstrap / connectivity self-check.

Reads the MySQL password from the MYSQL_PASSWORD environment variable (or
the .env file loaded by pydantic-settings), then creates the metadata tables
if missing.

Usage:
    python -m scripts.init_db
"""

from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.db.session import get_engine, init_schema


async def main() -> None:
    engine = get_engine()  # getpass prompt happens here
    async with engine.connect() as conn:
        version = (await conn.execute(text("SELECT VERSION()"))).scalar()
    print(f"[init_db] connected, MySQL version: {version}")
    await init_schema()
    print("[init_db] metadata tables ready (documents / tags / document_tags)")


if __name__ == "__main__":
    asyncio.run(main())
