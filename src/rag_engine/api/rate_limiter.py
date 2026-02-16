"""In-memory sliding window rate limiter."""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Depends, HTTPException, Request, status

from rag_engine.api.auth import verify_token
from rag_engine.observability.logger import get_logger

logger = get_logger("rate_limiter")


class SlidingWindowRateLimiter:
    """Simple in-memory sliding window rate limiter.

    Tracks request timestamps per key within a 60-second window.
    """

    def __init__(self) -> None:
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str, max_requests: int, window_seconds: int = 60) -> bool:
        """Return True if request is allowed, False if rate-limited."""
        now = time.monotonic()
        cutoff = now - window_seconds

        # Prune old entries
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

        if len(self._requests[key]) >= max_requests:
            return False

        self._requests[key].append(now)
        return True


# Singleton instance
_rate_limiter = SlidingWindowRateLimiter()


async def rate_limit(
    request: Request,
    token_payload: dict = Depends(verify_token),
) -> dict:
    """FastAPI dependency: enforce rate limiting per authenticated user.

    Chains verify_token internally. Returns the token payload for downstream use.
    """
    settings = request.app.state.settings
    key = token_payload.get("sub", "anonymous")

    if not _rate_limiter.check(key, settings.rate_limit_requests_per_minute):
        logger.warning("rate_limited", key=key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": "60"},
        )

    return token_payload
