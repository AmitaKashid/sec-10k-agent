"""redact_secrets() — strip DB credentials / secrets from strings before
they hit the log stream (CWE-532). Applied to str(exc) at the
startup.db.unreachable and shutdown.checkpointer.error log sites.
"""

from __future__ import annotations

from app.observability.logging import redact_secrets


def test_redacts_dsn_password_keeps_rest():
    raw = (
        'connection failed: connection to server at "db.host" (1.2.3.4), '
        "port 5432 failed: password authentication failed; "
        "url=postgresql+asyncpg://sec_agent:s3cr3t-pw@db.host:5432/sec_agent"
    )
    out = redact_secrets(raw)
    assert "s3cr3t-pw" not in out
    assert "sec_agent:***@db.host:5432/sec_agent" in out
    # Non-secret context is preserved for debuggability.
    assert "port 5432 failed" in out
    assert "db.host" in out


def test_redacts_key_value_secrets():
    assert redact_secrets("password=hunter2 host=x") == "password=*** host=x"
    assert redact_secrets('PWD="a b c"; db=main') == "PWD=***; db=main"
    assert "tok" not in redact_secrets("api_key=abc123def")
    assert redact_secrets("api_key=abc123def") == "api_key=***"


def test_clean_string_passes_through_unchanged():
    msg = "connection timed out after 5s (no credentials here)"
    assert redact_secrets(msg) == msg


def test_empty_and_none_safe():
    assert redact_secrets("") == ""
    # str(exc) is always a str; guard the falsy branch anyway.
    assert redact_secrets(None) is None  # type: ignore[arg-type]


def test_username_without_password_not_mangled():
    # asyncpg's 'failed for user "sec_agent"' has no password to strip;
    # must not be corrupted.
    msg = 'password authentication failed for user "sec_agent"'
    assert redact_secrets(msg) == msg
