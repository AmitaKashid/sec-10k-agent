"""Structured JSON logging for the backend.

Single entry point: `configure_logging()` is called once at FastAPI app
construction. After that, any `logging.getLogger(__name__).info("msg",
extra={...})` is emitted as one JSON line on stdout, automatically tagged
with `request_id` and `node` from contextvars when set.

Design notes
------------
- Stdout only (Railway captures it). No file handlers, no rotation.
- ContextVars (`request_id_var`, `node_var`) propagate across `await`
  boundaries, so a single request emits a coherent trace without every
  call site having to pass an ID through.
- `extra` fields are merged into the JSON payload. Reserved logging
  attributes (msg, args, levelname, …) are filtered out.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from contextvars import ContextVar
from typing import Any

# Redaction for log values that may contain a connection string or
# secret. DB drivers (asyncpg/psycopg/SQLAlchemy) routinely embed the
# full DSN — including the password — in str(exc); logging that raw is
# CWE-532. We scrub the credential but keep the rest of the message so
# the log line stays debuggable ("why" without the secret).
_DSN_USERINFO_RE = re.compile(
    r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)(?P<user>[^:/?#@\s]+):(?P<pw>[^@/?#\s]+)@"
)
_KV_SECRET_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key)\s*=\s*"
    r"('[^']*'|\"[^\"]*\"|[^\s;,&]+)"
)


def redact_secrets(text: str) -> str:
    """Strip DSN userinfo passwords and `key=secret` pairs from a string
    before it is logged. Returns the input unchanged when there is
    nothing to redact, so clean messages keep full fidelity."""
    if not text:
        return text
    text = _DSN_USERINFO_RE.sub(lambda m: f"{m.group('scheme')}{m.group('user')}:***@", text)
    text = _KV_SECRET_RE.sub(lambda m: f"{m.group(1)}=***", text)
    return text


request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
node_var: ContextVar[str | None] = ContextVar("node", default=None)


# Standard LogRecord attributes — everything else in record.__dict__ is
# treated as a custom field added via `extra=`.
_RESERVED_LOG_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        node = node_var.get()
        if node:
            payload["node"] = node

        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_ATTRS or key.startswith("_"):
                continue
            payload[key] = _safe(value)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def _safe(value: Any) -> Any:
    """Coerce non-JSON-native values to something serialisable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return str(value)


_LEVEL_COLORS = {
    "DEBUG": "\x1b[36m",  # cyan
    "INFO": "\x1b[32m",  # green
    "WARNING": "\x1b[33m",  # yellow
    "ERROR": "\x1b[31m",  # red
    "CRITICAL": "\x1b[1;31m",
}
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"

# Fields that are noisy in pretty mode — kept in JSON, hidden from terminal.
_PRETTY_HIDE = {"trace_events", "delta_keys", "color_message"}


class PrettyFormatter(logging.Formatter):
    """Compact, colorised single-line format for local dev terminals."""

    def __init__(self, *, color: bool = True) -> None:
        super().__init__()
        self.color = color

    def _c(self, code: str, text: str) -> str:
        return f"{code}{text}{_RESET}" if self.color else text

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = record.levelname
        level_str = self._c(_LEVEL_COLORS.get(level, ""), f"{level:<5}")

        rid = request_id_var.get()
        node = node_var.get()
        prefix_bits = []
        if rid:
            prefix_bits.append(self._c(_DIM, rid[:8]))
        if node:
            prefix_bits.append(self._c(_BOLD, node))
        prefix = " ".join(prefix_bits)

        msg = record.getMessage()

        extras: list[str] = []
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_ATTRS or key.startswith("_"):
                continue
            if key in _PRETTY_HIDE:
                continue
            v = _safe(value)
            if isinstance(v, (dict, list)):
                v = json.dumps(v, default=str, ensure_ascii=False)
            extras.append(f"{self._c(_DIM, key + '=')}{v}")

        parts = [self._c(_DIM, ts), level_str]
        if prefix:
            parts.append(prefix)
        parts.append(msg)
        if extras:
            parts.append(self._c(_DIM, "·") + " " + " ".join(extras))

        line = " ".join(parts)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


_CONFIGURED = False


def _pick_formatter() -> logging.Formatter:
    """Choose formatter from $LOG_FORMAT, defaulting to pretty on a TTY."""
    fmt = os.environ.get("LOG_FORMAT", "").lower()
    if fmt == "json":
        return JsonFormatter()
    if fmt == "pretty":
        return PrettyFormatter(color=sys.stdout.isatty())
    if sys.stdout.isatty():
        return PrettyFormatter(color=True)
    return JsonFormatter()


def configure_logging(level: str | int = "INFO") -> None:
    """Install stdout handler on the root logger. Idempotent.

    Format is JSON in production (default when stdout is not a TTY) and
    a compact colorised single-line layout in local dev. Override with
    LOG_FORMAT=pretty|json.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_pick_formatter())

    root = logging.getLogger()
    # Replace any default handlers (e.g. uvicorn's) so we don't double-print.
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down very chatty loggers but keep them available at DEBUG.
    for noisy in ("httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Uvicorn ships its own handlers; route them through ours.
    for uv in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(uv)
        lg.handlers.clear()
        lg.propagate = True

    _CONFIGURED = True


__all__ = [
    "configure_logging",
    "redact_secrets",
    "request_id_var",
    "node_var",
    "JsonFormatter",
    "PrettyFormatter",
]
