"""Tests for JWT auth and rate limiting."""

from __future__ import annotations

import time

import jwt
import pytest

from rag_engine.api.rate_limiter import SlidingWindowRateLimiter


def test_jwt_encode_decode():
    secret = "test-secret"
    payload = {"sub": "test-key", "iat": int(time.time()), "exp": int(time.time()) + 3600}
    token = jwt.encode(payload, secret, algorithm="HS256")
    decoded = jwt.decode(token, secret, algorithms=["HS256"])
    assert decoded["sub"] == "test-key"


def test_jwt_expired():
    secret = "test-secret"
    payload = {"sub": "test-key", "iat": int(time.time()) - 7200, "exp": int(time.time()) - 3600}
    token = jwt.encode(payload, secret, algorithm="HS256")
    with pytest.raises(jwt.ExpiredSignatureError):
        jwt.decode(token, secret, algorithms=["HS256"])


def test_jwt_invalid_secret():
    secret = "test-secret"
    payload = {"sub": "test-key", "iat": int(time.time()), "exp": int(time.time()) + 3600}
    token = jwt.encode(payload, secret, algorithm="HS256")
    with pytest.raises(jwt.InvalidSignatureError):
        jwt.decode(token, "wrong-secret", algorithms=["HS256"])


def test_rate_limiter_allows():
    limiter = SlidingWindowRateLimiter()
    for _ in range(5):
        assert limiter.check("user1", max_requests=5) is True
    assert limiter.check("user1", max_requests=5) is False


def test_rate_limiter_separate_keys():
    limiter = SlidingWindowRateLimiter()
    for _ in range(5):
        limiter.check("user1", max_requests=5)
    # user1 is exhausted
    assert limiter.check("user1", max_requests=5) is False
    # user2 is fresh
    assert limiter.check("user2", max_requests=5) is True


def test_rate_limiter_empty_key():
    limiter = SlidingWindowRateLimiter()
    assert limiter.check("", max_requests=1) is True
    assert limiter.check("", max_requests=1) is False
