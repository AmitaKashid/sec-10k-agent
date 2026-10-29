"""Driver-compatible checkpoint URLs and authenticated thread isolation."""
from __future__ import annotations

import hashlib
import json
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def checkpoint_conn_string(database_url: str) -> str:
    """Translate SQLAlchemy/asyncpg URL options into libpq options."""
    parts = urlsplit(database_url)
    if parts.scheme not in ("postgresql+asyncpg", "postgresql", "postgres"):
        raise ValueError("Unsupported checkpoint database scheme")
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    ssl = query.pop("ssl", None)
    if ssl is not None:
        if ssl not in ("disable", "allow", "prefer", "require", "verify-ca", "verify-full"):
            raise ValueError("Unsupported checkpoint SSL mode")
        if "sslmode" in query and query["sslmode"] != ssl:
            raise ValueError("Conflicting checkpoint SSL modes")
        query["sslmode"] = ssl
    return urlunsplit(parts._replace(scheme="postgresql", query=urlencode(query)))


def checkpoint_key(user_id: str, thread_id: str) -> str:
    """Private storage key; public thread IDs cannot select another user's state."""
    if not user_id or not thread_id:
        raise ValueError("User and thread identifiers are required")
    encoded = json.dumps([user_id, thread_id], separators=(",", ":")).encode()
    return "user:" + hashlib.sha256(encoded).hexdigest()
