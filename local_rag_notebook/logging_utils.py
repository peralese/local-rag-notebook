from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional


class _PlainFormatter(logging.Formatter):
    """Human-friendly single-line formatter (to stderr)."""

    default_fmt = "%(levelname)s %(name)s - %(message)s"
    verbose_fmt = (
        "%(asctime)s %(levelname)s %(name)s [%(process)d:%(threadName)s] "
        "%(filename)s:%(lineno)d - %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S"

    def __init__(self, debug: bool = False) -> None:
        fmt = self.verbose_fmt if debug else self.default_fmt
        super().__init__(fmt=fmt, datefmt=self.datefmt)


class _JsonFormatter(logging.Formatter):
    """JSON lines formatter (each record is one JSON object)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Optional context
        payload["process"] = record.process
        payload["thread"] = record.threadName
        payload["file"] = record.filename
        payload["line"] = record.lineno
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _coerce_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        upper = level.upper()
        if upper in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
            return getattr(logging, upper)
        # Allow "1/2/3/4/5" style
        if upper.isdigit():
            return int(upper)
    return logging.INFO


def setup_logging(
    level: str | int | None = None,
    json_logs: bool = False,
    propagate_root: bool = False,
) -> None:
    """
    Configure root logging once for the process.

    Args:
        level: Logging level (e.g., "DEBUG", "INFO"). Defaults to INFO.
        json_logs: If True, emit JSON lines to stderr; otherwise plain text.
        propagate_root: If True, let logs bubble to any parent handlers.
    """
    # Derive final level (CLI flag takes precedence, then env, then default).
    env_level = os.getenv("LOG_LEVEL")
    final_level = _coerce_level(level or env_level or logging.INFO)

    root = logging.getLogger()
    root.setLevel(final_level)

    # Clear existing handlers only once to avoid duplicate logs in REPL/tests
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stderr)
    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_PlainFormatter(debug=(final_level <= logging.DEBUG)))
    root.addHandler(handler)
    root.propagate = propagate_root

    # Quiet some noisy third-party loggers by default
    for noisy in ("urllib3", "httpx", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(max(final_level, logging.WARNING))
