"""Structured logging.

Pricing failures are hard to reconstruct from a bare stack trace: what matters
is the trade that caused them. These helpers attach the inputs to the log
record so a failed quote can be replayed exactly.

Emits JSON when ``OTCPRICER_LOG_FORMAT=json`` (for log shipping), and a compact
human-readable line otherwise (for local development).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

__all__ = ["configure_logging", "get_logger", "JsonFormatter"]

#: Attributes present on every LogRecord; anything else was added by the caller
#: via ``extra=`` and therefore belongs in the structured payload.
_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {
    "message",
    "asctime",
    "taskName",
}


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    """Fields the caller attached via ``extra=``."""
    return {k: v for k, v in vars(record).items() if k not in _RESERVED}


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extras = _extras(record)
        if extras:
            payload["ctx"] = extras
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _TextFormatter(logging.Formatter):
    """Human-readable line with any ``extra=`` context appended."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = _extras(record)
        if extras:
            rendered = " ".join(f"{k}={v!r}" for k, v in sorted(extras.items()))
            return f"{base}  [{rendered}]"
        return base


def configure_logging(level: str | int | None = None, *, fmt: str | None = None) -> None:
    """Install a single stderr handler on the ``otcpricer`` logger.

    Idempotent: calling it twice does not double every line, which matters
    because the Flask dev server reloader imports the app module twice.

    Args:
        level: Log level; defaults to ``$OTCPRICER_LOG_LEVEL`` or ``INFO``.
        fmt: ``"json"`` or ``"text"``; defaults to ``$OTCPRICER_LOG_FORMAT``
            or ``"text"``.
    """
    resolved_level = level if level is not None else os.getenv("OTCPRICER_LOG_LEVEL", "INFO")
    resolved_fmt = (fmt or os.getenv("OTCPRICER_LOG_FORMAT") or "text").lower()

    root = logging.getLogger("otcpricer")
    root.setLevel(resolved_level)
    root.propagate = False

    for existing in list(root.handlers):
        root.removeHandler(existing)

    handler = logging.StreamHandler(stream=sys.stderr)
    if resolved_fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            _TextFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``otcpricer`` namespace.

    Pass ``__name__``; a module-qualified name already starting with
    ``otcpricer`` is used as-is.
    """
    if name.startswith("otcpricer"):
        return logging.getLogger(name)
    return logging.getLogger(f"otcpricer.{name}")
