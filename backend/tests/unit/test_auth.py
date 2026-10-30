"""Unit tests for app.api.auth.

Real RSA keypair + real PyJWT sign/verify path. The JWKS HTTP fetch is
stubbed out by patching `_get_jwk_client` so the test does not touch the
network or depend on Clerk.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, HTTPException, Request, status

from app.api import auth as auth_module
from app.api.auth import Identity, _verify_token, resolve_identity

# --- helpers -------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_keypair() -> tuple[Any, Any]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


@pytest.fixture
def patch_jwks(monkeypatch, rsa_keypair):
    """Replace the module-level JWKS client with a stub that returns our
    public key for any token. Resets the cached client too."""
    _, public_pem = rsa_keypair
    fake_signing_key = MagicMock()
    fake_signing_key.key = public_pem
    fake_client = MagicMock()
    fake_client.get_signing_key_from_jwt.return_value = fake_signing_key
    monkeypatch.setattr(auth_module, "_jwk_client", None)
    monkeypatch.setattr(auth_module, "_get_jwk_client", lambda: fake_client)
    monkeypatch.setattr(auth_module.settings, "CLERK_AUTHORIZED_PARTIES", [])
    monkeypatch.setattr(auth_module.settings, "RATE_LIMIT_TRUST_XFF", False)
    # Set a non-empty issuer so jwt.decode's iss check runs against it.
    monkeypatch.setattr(auth_module.settings, "CLERK_ISSUER", "https://test.clerk.dev")
    monkeypatch.setattr(
        auth_module.settings,
        "CLERK_JWKS_URL",
        "https://test.clerk.dev/.well-known/jwks.json",
    )


def _make_token(private_pem: bytes, **claims) -> str:
    now = int(time.time())
    payload = {
        "iss": "https://test.clerk.dev",
        "sub": "user_abc123",
        "iat": now,
        "exp": now + 60,
        **claims,
    }
    return jwt.encode(payload, private_pem, algorithm="RS256")


def _make_request(headers: dict[str, str] | None = None, client_ip: str = "1.2.3.4") -> Request:
    """Minimal ASGI scope shaped Request for the dependency."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/ask",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": (client_ip, 12345),
        "app": FastAPI(),  # so request.state works
    }
    return Request(scope)


# --- _verify_token -------------------------------------------------------


def test_verify_token_returns_sub_on_valid_token(rsa_keypair, patch_jwks):
    private_pem, _ = rsa_keypair
    token = _make_token(private_pem, sub="user_xyz")
    assert _verify_token(token) == "user_xyz"


def test_verify_token_rejects_bad_signature(rsa_keypair, patch_jwks):
    # Sign with a DIFFERENT key — verification key (patched) won't match.
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other_pem = other_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    bad_token = _make_token(other_pem)
    with pytest.raises(jwt.PyJWTError):
        _verify_token(bad_token)


def test_verify_token_rejects_expired(rsa_keypair, patch_jwks):
    private_pem, _ = rsa_keypair
    now = int(time.time())
    expired = _make_token(private_pem, iat=now - 3600, exp=now - 1800)
    with pytest.raises(jwt.ExpiredSignatureError):
        _verify_token(expired)


def test_verify_token_rejects_wrong_issuer(rsa_keypair, patch_jwks):
    private_pem, _ = rsa_keypair
    token = _make_token(private_pem, iss="https://evil.example.com")
    with pytest.raises(jwt.InvalidIssuerError):
        _verify_token(token)


def test_verify_token_rejects_missing_sub(rsa_keypair, patch_jwks, monkeypatch):
    private_pem, _ = rsa_keypair
    # Build a token without sub manually (jwt.encode allows arbitrary payload).
    now = int(time.time())
    token = jwt.encode(
        {"iss": "https://test.clerk.dev", "iat": now, "exp": now + 60},
        private_pem,
        algorithm="RS256",
    )
    with pytest.raises(jwt.PyJWTError):
        _verify_token(token)


# --- resolve_identity ----------------------------------------------------


async def test_resolve_identity_anonymous_when_no_token(patch_jwks):
    req = _make_request()
    identity = await resolve_identity(req)
    assert identity.tier == "anon"
    assert identity.key == "1.2.3.4"
    assert req.state.identity == identity
    assert identity.bucket_key == "anon:1.2.3.4"


async def test_resolve_identity_user_when_valid_bearer(rsa_keypair, patch_jwks):
    private_pem, _ = rsa_keypair
    token = _make_token(private_pem, sub="user_signed_in")
    req = _make_request(headers={"authorization": f"Bearer {token}"})
    identity = await resolve_identity(req)
    assert identity.tier == "user"
    assert identity.key == "user_signed_in"
    assert identity.bucket_key == "user:user_signed_in"


async def test_resolve_identity_401_on_invalid_bearer(patch_jwks):
    req = _make_request(headers={"authorization": "Bearer junk.not.a.jwt"})
    with pytest.raises(HTTPException) as exc:
        await resolve_identity(req)
    assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED


async def test_resolve_identity_401_when_jwks_unconfigured(monkeypatch, rsa_keypair):
    # No patch_jwks fixture here — start from a clean unconfigured state.
    private_pem, _ = rsa_keypair
    monkeypatch.setattr(auth_module, "_jwk_client", None)
    monkeypatch.setattr(auth_module.settings, "CLERK_JWKS_URL", "")
    monkeypatch.setattr(auth_module.settings, "CLERK_ISSUER", "")
    token = _make_token(private_pem)
    req = _make_request(headers={"authorization": f"Bearer {token}"})
    with pytest.raises(HTTPException) as exc:
        await resolve_identity(req)
    assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED


async def test_resolve_identity_ignores_non_bearer_auth_header(patch_jwks):
    req = _make_request(headers={"authorization": "Basic dXNlcjpwYXNz"})
    identity = await resolve_identity(req)
    assert identity.tier == "anon"


async def test_resolve_identity_rejects_oversized_auth_header(patch_jwks):
    # Oversized Authorization must 401, not silently downgrade to anon.
    # Silent downgrade would let a token holder pad their header to
    # escape per-user tracking and fall into the anon IP bucket.
    giant = "Bearer " + ("x" * (10 * 1024))
    req = _make_request(headers={"authorization": giant})
    with pytest.raises(HTTPException) as exc:
        await resolve_identity(req)
    assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED


# --- proxy-aware client IP -----------------------------------------------


async def test_anon_ip_ignores_xff_when_trust_disabled(patch_jwks, monkeypatch):
    monkeypatch.setattr(auth_module.settings, "RATE_LIMIT_TRUST_XFF", False)
    req = _make_request(headers={"x-forwarded-for": "203.0.113.5"}, client_ip="10.0.0.1")
    identity = await resolve_identity(req)
    # Trust disabled → still uses socket peer, not XFF.
    assert identity.key == "10.0.0.1"


async def test_anon_ip_reads_xff_when_trust_enabled(patch_jwks, monkeypatch):
    monkeypatch.setattr(auth_module.settings, "RATE_LIMIT_TRUST_XFF", True)
    req = _make_request(headers={"x-forwarded-for": "203.0.113.5, 10.0.0.7"}, client_ip="10.0.0.1")
    identity = await resolve_identity(req)
    # Leftmost = originating client per RFC 7239 convention.
    assert identity.key == "203.0.113.5"


async def test_anon_ip_falls_back_when_xff_missing_and_trust_enabled(patch_jwks, monkeypatch):
    monkeypatch.setattr(auth_module.settings, "RATE_LIMIT_TRUST_XFF", True)
    req = _make_request(client_ip="10.0.0.1")
    identity = await resolve_identity(req)
    assert identity.key == "10.0.0.1"


async def test_verify_token_rejects_wrong_azp_when_authorized_parties_set(
    rsa_keypair, patch_jwks, monkeypatch
):
    private_pem, _ = rsa_keypair
    monkeypatch.setattr(
        auth_module.settings,
        "CLERK_AUTHORIZED_PARTIES",
        ["https://app.example.com"],
    )
    token = _make_token(private_pem, azp="https://evil.example.com")
    with pytest.raises(jwt.PyJWTError):
        _verify_token(token)


async def test_verify_token_accepts_matching_azp(rsa_keypair, patch_jwks, monkeypatch):
    private_pem, _ = rsa_keypair
    monkeypatch.setattr(
        auth_module.settings,
        "CLERK_AUTHORIZED_PARTIES",
        ["https://app.example.com"],
    )
    token = _make_token(private_pem, azp="https://app.example.com")
    assert _verify_token(token) == "user_abc123"


async def test_verify_token_skips_azp_when_allowlist_empty(rsa_keypair, patch_jwks, monkeypatch):
    # Default behavior in dev: no azp allowlist configured → don't enforce.
    private_pem, _ = rsa_keypair
    monkeypatch.setattr(auth_module.settings, "CLERK_AUTHORIZED_PARTIES", [])
    token = _make_token(private_pem, azp="https://anything.example.com")
    assert _verify_token(token) == "user_abc123"


def test_jwk_client_fails_fast_when_issuer_missing(monkeypatch):
    monkeypatch.setattr(auth_module, "_jwk_client", None)
    monkeypatch.setattr(
        auth_module.settings,
        "CLERK_JWKS_URL",
        "https://test.clerk.dev/.well-known/jwks.json",
    )
    monkeypatch.setattr(auth_module.settings, "CLERK_ISSUER", "")
    with pytest.raises(RuntimeError, match="CLERK_ISSUER"):
        auth_module._get_jwk_client()


def test_jwk_client_rejects_non_https_jwks_url(monkeypatch):
    monkeypatch.setattr(auth_module, "_jwk_client", None)
    monkeypatch.setattr(
        auth_module.settings,
        "CLERK_JWKS_URL",
        "http://test.clerk.dev/.well-known/jwks.json",
    )
    monkeypatch.setattr(auth_module.settings, "CLERK_ISSUER", "https://test.clerk.dev")
    with pytest.raises(RuntimeError, match="https"):
        auth_module._get_jwk_client()


async def test_anon_ip_prefers_cf_connecting_ip_over_xff(patch_jwks, monkeypatch):
    monkeypatch.setattr(auth_module.settings, "RATE_LIMIT_TRUST_XFF", True)
    req = _make_request(
        headers={
            "cf-connecting-ip": "198.51.100.42",
            "x-forwarded-for": "203.0.113.5, 10.0.0.7",
        },
        client_ip="10.0.0.1",
    )
    identity = await resolve_identity(req)
    # CDN-injected single-value header is unspoofable through the origin
    # in a way XFF is not — prefer it.
    assert identity.key == "198.51.100.42"


async def test_anon_ip_prefers_x_real_ip_over_xff(patch_jwks, monkeypatch):
    monkeypatch.setattr(auth_module.settings, "RATE_LIMIT_TRUST_XFF", True)
    req = _make_request(
        headers={
            "x-real-ip": "198.51.100.99",
            "x-forwarded-for": "203.0.113.5",
        },
        client_ip="10.0.0.1",
    )
    identity = await resolve_identity(req)
    assert identity.key == "198.51.100.99"


def test_identity_bucket_key_separates_tiers():
    anon = Identity(tier="anon", key="1.2.3.4")
    # Even if the IP from a signed-out request matches a sub from a
    # signed-in user (artificial but illustrative), tiers must not bleed.
    user = Identity(tier="user", key="1.2.3.4")
    assert anon.bucket_key != user.bucket_key
