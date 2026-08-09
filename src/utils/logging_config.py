"""Structured logging setup, shared by every component in the pipeline.

Emits JSON lines by default (consumed by the Monitoring stack in Milestone 11)
with a human-readable fallback for local development.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Renders each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Anything passed via logger.info(..., extra={...}) rides along in the JSON line
        reserved = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}
        for key, value in vars(record).items():
            if key not in reserved and key not in payload:
                try:
                    json.dumps(value)  # only include JSON-serializable extras
                    payload[key] = value
                except TypeError:
                    payload[key] = str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str, *, json_logs: bool = True, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Safe to call repeatedly (idempotent handler setup)."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured — don't stack duplicate handlers
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(stream=sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
