"""Per-identity token budget for /ask.

Sliding-window in-process bucket keyed by `Identity.bucket_key`. Tracks
LLM token usage over the trailing 60s. See
docs/auth_and_rate_limit_plan.md §Token budget.

Single-replica only. Cross-replica accuracy (Redis/Upstash) is
deliberately out of scope until we scale.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal, NamedTuple

from fastapi.responses import JSONResponse

from app.api.auth import Identity
from app.settings import settings

WINDOW_SECONDS = 60


class TierLimits(NamedTuple):
    reqs_per_min: int
    tokens_per_min: int
    max_input_tokens: int


def tier_limits(tier: str) -> TierLimits:
    if tier == "user":
        return TierLimits(
            reqs_per_min=settings.ASK_USER_REQS_PER_MIN,
            tokens_per_min=settings.ASK_USER_TOKENS_PER_MIN,
            max_input_tokens=settings.ASK_USER_MAX_INPUT_TOKENS,
        )
    return TierLimits(
        reqs_per_min=settings.ASK_ANON_REQS_PER_MIN,
        tokens_per_min=settings.ASK_ANON_TOKENS_PER_MIN,
        max_input_tokens=settings.ASK_ANON_MAX_INPUT_TOKENS,
    )


@dataclass(frozen=True)
class Reservation:
    bucket_key: str
    entry_id: int
    enabled: bool  # False = limiter disabled, finalize is a no-op


class TokenBudget:
    """Sliding-window token bucket. Thread-safe via a single asyncio.Lock —
    contention is negligible at expected QPS."""

    def __init__(self, *, time_fn=time.monotonic):
        # bucket_key -> deque[(entry_id, ts, tokens)]
        self._buckets: dict[str, deque[tuple[int, float, int]]] = {}
        self._lock = asyncio.Lock()
        self._next_id = 0
        self._time = time_fn

    def _evict(self, dq: deque[tuple[int, float, int]], now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        while dq and dq[0][1] < cutoff:
            dq.popleft()

    def _used(self, dq: deque[tuple[int, float, int]]) -> int:
        return sum(tokens for _, _, tokens in dq)

    async def check_and_reserve(
        self,
        identity: Identity,
        est_tokens: int,
    ) -> tuple[Reservation | None, int, int]:
        """Returns (reservation_or_None, used_after_reserve, budget).

        On success: reservation is non-None, used reflects the new total.
        On reject: reservation is None, used reflects the *current* total
            (without the rejected reservation) — caller uses it in the
            429 response body. budget is the tier cap.
        """
        if not settings.ASK_TOKEN_LIMITER_ENABLED:
            return Reservation(bucket_key="-", entry_id=-1, enabled=False), 0, 0

        limits = tier_limits(identity.tier)
        budget = limits.tokens_per_min
        bucket_key = identity.bucket_key

        async with self._lock:
            dq = self._buckets.setdefault(bucket_key, deque())
            now = self._time()
            self._evict(dq, now)
            used = self._used(dq)
            if used + est_tokens > budget:
                return None, used, budget
            self._next_id += 1
            entry_id = self._next_id
            dq.append((entry_id, now, est_tokens))
            return (
                Reservation(bucket_key=bucket_key, entry_id=entry_id, enabled=True),
                used + est_tokens,
                budget,
            )

    async def finalize(self, reservation: Reservation, actual_tokens: int) -> None:
        """Reconcile the reservation against real usage. If the entry was
        evicted (>60s elapsed before finalize), append a fresh entry so
        the actual cost still counts toward the window."""
        if not reservation.enabled:
            return
        async with self._lock:
            dq = self._buckets.get(reservation.bucket_key)
            now = self._time()
            if dq is None:
                # Bucket gone entirely — record as a new entry.
                dq = self._buckets.setdefault(reservation.bucket_key, deque())
                dq.append((reservation.entry_id, now, actual_tokens))
                return
            self._evict(dq, now)
            for i, (eid, ts, _) in enumerate(dq):
                if eid == reservation.entry_id:
                    dq[i] = (eid, ts, actual_tokens)
                    return
            # Reservation evicted before finalize — append fresh.
            dq.append((reservation.entry_id, now, actual_tokens))

    async def used(self, identity: Identity) -> int:
        """Diagnostic: current usage for an identity. Evicts stale entries
        as a side effect."""
        async with self._lock:
            dq = self._buckets.get(identity.bucket_key)
            if dq is None:
                return 0
            self._evict(dq, self._time())
            return self._used(dq)


class RequestBudget:
    """Per-tier request-count limiter, sliding 60s window. Same locking
    model as TokenBudget. Kept separate so the two caps reconcile
    independently — exhausting the request count must not lock you out
    of the token budget and vice versa."""

    def __init__(self, *, time_fn=time.monotonic):
        self._buckets: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()
        self._time = time_fn

    async def check_and_consume(self, identity: Identity) -> tuple[bool, int, int]:
        """Returns (allowed, used_after, budget). On allowed=True, the hit
        is recorded. On False, nothing is recorded."""
        if not settings.ASK_TOKEN_LIMITER_ENABLED:
            return True, 0, 0
        limits = tier_limits(identity.tier)
        budget = limits.reqs_per_min
        bucket_key = identity.bucket_key
        async with self._lock:
            dq = self._buckets.setdefault(bucket_key, deque())
            now = self._time()
            cutoff = now - WINDOW_SECONDS
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= budget:
                return False, len(dq), budget
            dq.append(now)
            return True, len(dq), budget

    async def release(self, identity: Identity) -> None:
        """Refund the most recently consumed slot for this identity. Used
        when a subsequent check (e.g. token budget) rejects the request —
        a rejected request should not burn a request-count slot."""
        if not settings.ASK_TOKEN_LIMITER_ENABLED:
            return
        async with self._lock:
            dq = self._buckets.get(identity.bucket_key)
            if dq:
                dq.pop()


# Module-level singletons. Tests can construct their own instances with a
# fake clock; production routes use these.
BUDGET = TokenBudget()
REQUESTS = RequestBudget()


# --- response helpers ----------------------------------------------------


LimitReason = Literal[
    "anonymous_limit",
    "authed_limit",
    "input_too_large",
    "anonymous_req_limit",
    "authed_req_limit",
    "anonymous_lifetime_limit",
    "anon_ticker_not_allowed",
]


def limit_response(
    *,
    status_code: int,
    detail: str,
    reason: LimitReason,
    identity: Identity,
    used: int,
    budget: int,
) -> JSONResponse:
    """Shared JSON shape for 413/429 from the limiter layer.

    `auth_required` is True iff the caller is anonymous — the frontend uses
    it to decide between "open sign-in modal" and "show toast"."""
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": detail,
            "reason": reason,
            "auth_required": identity.tier == "anon",
            "limit": {"used": used, "budget": budget, "window_s": WINDOW_SECONDS},
        },
        headers={"Retry-After": str(WINDOW_SECONDS)},
    )


__all__ = [
    "BUDGET",
    "REQUESTS",
    "RequestBudget",
    "Reservation",
    "TierLimits",
    "TokenBudget",
    "WINDOW_SECONDS",
    "limit_response",
    "tier_limits",
]
