"""SQLAlchemy ORM models for document metadata (MySQL)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all metadata tables."""


class Document(Base):
    """One uploaded document (identity = normalized file name + extension)."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_key: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    ext: Mapped[str] = mapped_column(String(16))
    modality: Mapped[str] = mapped_column(String(32), default="text")
    file_path: Mapped[str] = mapped_column(String(512), default="")
    parsed_text: Mapped[str | None] = mapped_column(MEDIUMTEXT, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    created_by: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


class Tag(Base):
    """A document category tag (LLM-suggested or user-defined)."""

    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    source: Mapped[str] = mapped_column(String(16), default="llm")  # llm / custom
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class DocumentTag(Base):
    """Many-to-many link between documents and tags."""

    __tablename__ = "document_tags"
    __table_args__ = (UniqueConstraint("doc_key", "tag_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_key: Mapped[str] = mapped_column(String(32), index=True)
    tag_id: Mapped[int] = mapped_column(BigInteger, index=True)
