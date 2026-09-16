"""Full-chain audit logging for every agent/tool hop.

Each audit record is one JSON line with a shared trace_id so the whole
chain (user -> assistant -> intent -> agent -> tool -> response) can be
reconstructed for compliance review.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.security.masking import mask_sensitive

_lock = threading.Lock()


def new_trace_id() -> str:
    """Generate a unique trace id for one user turn."""
    return uuid.uuid4().hex


class AuditLogger:
    """Append-only JSONL audit sink (fan-out to SIEM in production)."""

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or get_settings().audit_log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        trace_id: str,
        actor: str,
        action: str,
        detail: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> None:
        """Write one audit record. Sensitive fields are masked first."""
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "trace_id": trace_id,
            "session_id": session_id,
            "actor": actor,
            "action": action,
            "detail": mask_sensitive(detail or {}),
        }
        line = json.dumps(record, ensure_ascii=False)
        with _lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Process-wide singleton audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
