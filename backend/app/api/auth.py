"""Identity resolution for /ask. See docs/auth_and_rate_limit_plan.md §Identity.

A `resolve_identity` FastAPI dependency that:

  * Verifies a Clerk-issued JWT in the `Authorization: Bearer <token>` header
    against Clerk's JWKS (cached in-process by PyJWKClient). On success
    returns `Identity(tier="user", key=<sub>)`.
  * On a present-but-invalid token, raises 401. We deliberately do NOT
    silently downgrade to anonymous — that would mask a misconfigured
    frontend.
  * On a missing token, returns `Identity(tier="anon", key=<remote_ip>)`.

The resolved identity is also stashed on `request.state.identity` so the
slowapi key function and the (later) token-budget limiter can read it
without re-running verification.
"""

from __future__ import annotations

import logging
from typing import Literal
from urllib.parse import urlparse

import jwt
from fastapi import HTTPException, Request, status
from pydantic import BaseModel
from slowapi.util import get_remote_address

from app.settings import settings

logger = logging.getLogger(__name__)


class Identity(BaseModel):
    tier: Literal["anon", "user"]
    key: str  # clerk `sub` for user, remote IP for anon

    @property
    def bucket_key(self) -> str:
        """Composite key used by rate limiters. Tier is part of the key so a
        signed-out request from an IP that *was* signed in does not inherit
        the user's prior usage."""
        return f"{self.tier}:{self.key}"


# Module-level JWKS client. PyJWKClient fetches lazily on first
# get_signing_key_from_jwt() call and caches keys in memory for the
# process lifetime. We construct it lazily (after settings are loaded)
# and only when a token is actually presented.
_jwk_client: jwt.PyJWKClient | None = None


def _get_jwk_client() -> jwt.PyJWKClient:
    global _jwk_client
    if _jwk_client is None:
        if not settings.CLERK_JWKS_URL:
            # Server misconfiguration — surface loudly rather than silently
            # downgrading the request. The 401 caller wraps this.
            raise RuntimeError("CLERK_JWKS_URL is not configured")
        if not settings.CLERK_ISSUER:
            # An empty issuer would cause PyJWT to skip `iss` validation
            # entirely (it treats `issuer=""` as "don't check"), turning
            # a configured-JWKS deployment into an unauthenticated one
            # for anyone who can mint an RSA-signed JWT served by the
            # JWKS URL. Hard-fail instead.
            raise RuntimeError("CLERK_JWKS_URL is set but CLERK_ISSUER is not")
        # Pin JWKS to https. Without this an env-poisoning or typo'd
        # value (http://, file://, internal host) would turn the JWKS
        # fetch into an SSRF / MitM vector — any RSA token signed by an
        # attacker-controlled JWKS would then be accepted.
        parsed = urlparse(settings.CLERK_JWKS_URL)
        if parsed.scheme != "https" or not parsed.hostname:
            raise RuntimeError("CLERK_JWKS_URL must be an https:// URL with a hostname")
        _jwk_client = jwt.PyJWKClient(settings.CLERK_JWKS_URL, cache_keys=True)
    return _jwk_client


def _verify_token(token: str) -> str:
    """Verify a Clerk JWT. Returns the `sub` claim. Raises jwt.PyJWTError
    on any verification failure.

    Hardening:
      * `iss` is required and pinned to `settings.CLERK_ISSUER` (PyJWT
        skips iss validation when `issuer=None`, so we hard-fail at
        startup if CLERK_JWKS_URL is set but CLERK_ISSUER is empty —
        see `_get_jwk_client`).
      * `azp` (authorized party) is required when
        `settings.CLERK_AUTHORIZED_PARTIES` is configured; rejects
        tokens minted for a different frontend origin even if they're
        signed by the same Clerk app's JWKS. Clerk's standard
        cross-app-reuse defense.
    """
    signing_key = _get_jwk_client().get_signing_key_from_jwt(token).key
    payload = jwt.decode(
        token,
        signing_key,
        algorithms=["RS256"],
        issuer=settings.CLERK_ISSUER,
        leeway=10,  # clock-skew tolerance, seconds
        options={"require": ["exp", "iat", "sub", "iss"]},
    )
    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        raise jwt.InvalidTokenError("missing or empty sub claim")

    authorized_parties = settings.CLERK_AUTHORIZED_PARTIES
    if authorized_parties:
        azp = payload.get("azp")
        if not isinstance(azp, str) or azp not in authorized_parties:
            raise jwt.InvalidTokenError("azp claim missing or not authorized")

    return sub


# Cap on Authorization header length. PyJWKClient parses the unverified
# JWT header to look up `kid`, so a multi-MB Bearer string forces real
# work before any signature check. Clerk session tokens are ~1-2 KB;
# 8 KB leaves generous headroom.
_MAX_AUTH_HEADER_BYTES = 8 * 1024


def _get_client_ip(request: Request) -> str:
    """Best-effort client IP for anon rate-limit keying.

    Behind a trusted proxy (Railway, Vercel), the socket peer is the
    proxy — `request.client.host` collapses every anon user onto one
    bucket. When `settings.RATE_LIMIT_TRUST_XFF` is True, read the
    leftmost entry of `X-Forwarded-For` (the originating client as
    seen by the proxy). Falls back to the socket peer when XFF is
    absent or trust is disabled.

    Security: only enable RATE_LIMIT_TRUST_XFF when ALL inbound
    traffic flows through a proxy that strips or replaces
    client-supplied XFF headers. Otherwise a caller can spoof the
    header to evade the per-IP cap.
    """
    if settings.RATE_LIMIT_TRUST_XFF:
        # Prefer single-value proxy-injected headers when present —
        # Cloudflare and Railway both inject `cf-connecting-ip` /
        # `x-real-ip`, which the client cannot forge through to the
        # origin. These are unspoofable in a way the multi-hop XFF
        # header is not.
        for header in ("cf-connecting-ip", "x-real-ip"):
            v = request.headers.get(header)
            if v:
                v = v.strip()
                if v:
                    return v
        xff = request.headers.get("x-forwarded-for")
        if xff:
            # Leftmost entry is the originating client. Trim whitespace
            # and reject empty results so we fall back cleanly.
            first = xff.split(",", 1)[0].strip()
            if first:
                return first
    return get_remote_address(request) or "unknown"


def _extract_bearer(request: Request) -> str | None:
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not header:
        return None
    if len(header) > _MAX_AUTH_HEADER_BYTES:
        # Reject loudly rather than silently downgrading to anon — a
        # silent downgrade lets a token holder pad their Authorization
        # header to escape per-user tracking and fall into the (often
        # stricter) anon IP bucket. CWE-287.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authorization header too large",
        )
    parts = header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


async def resolve_identity(request: Request) -> Identity:
    """FastAPI dependency. See module docstring."""
    token = _extract_bearer(request)
    if token is not None:
        try:
            sub = _verify_token(token)
        except jwt.PyJWTError as exc:
            logger.warning(
                "auth.token.invalid",
                extra={"error_type": type(exc).__name__},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid token",
            ) from exc
        except RuntimeError as exc:
            # CLERK_JWKS_URL missing — config error, not the user's fault.
            # Still 401 (not 500) so the response shape matches other auth
            # failures; the log line distinguishes them for ops.
            logger.error("auth.misconfigured: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="auth not configured",
            ) from exc
        identity = Identity(tier="user", key=sub)
    else:
        identity = Identity(tier="anon", key=_get_client_ip(request))

    # Stash on request.state so downstream code (rate limiter key func,
    # token budget) can read without re-verifying.
    request.state.identity = identity
    return identity


__all__ = ["Identity", "resolve_identity"]
