"""
tests/test_rate_limiter.py

Unit tests for gateway/rate_limiter.py — sliding-window rate limiting.

Uses fakeredis so no real Redis instance is required.

Test coverage goals:
  - Requests within the window limit are allowed.
  - The (max_requests + 1)th request raises RateLimitExceeded.
  - RateLimitExceeded carries the correct retry_after value.
  - Requests older than window_seconds are not counted (window slides).
  - Different keys have independent counters.
  - peek() returns the correct (used, remaining, reset_at) without consuming quota.
  - reset() clears the window for a key.
  - Concurrent requests from multiple coroutines do not exceed the limit
    (atomicity test via asyncio.gather).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from gateway.rate_limiter import RateLimiter, RateLimitExceeded, WindowConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def window_config() -> WindowConfig:
    """A tight window for fast tests: 5 requests per 10 seconds."""
    return WindowConfig(max_requests=5, window_seconds=10, key_prefix="test")


# ---------------------------------------------------------------------------
# Basic allow / deny
# ---------------------------------------------------------------------------

class TestBasicLimiting:
    """Core allow/deny behaviour."""

    @pytest.mark.asyncio
    async def test_requests_within_limit_are_allowed(
        self, rate_limiter, window_config
    ):
        """
        The first max_requests calls to check() should not raise.
        """
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_request_exceeding_limit_raises(self, rate_limiter, window_config):
        """
        The (max_requests + 1)th call to check() should raise RateLimitExceeded.
        """
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_has_retry_after(
        self, rate_limiter, window_config
    ):
        """
        RateLimitExceeded.retry_after should be a positive float indicating
        when the window resets.
        """
        # TODO: implement
        ...


# ---------------------------------------------------------------------------
# Window sliding
# ---------------------------------------------------------------------------

class TestWindowSliding:
    """Verify the sliding-window behaviour (old requests drop off)."""

    @pytest.mark.asyncio
    async def test_old_requests_not_counted(self, rate_limiter, window_config, freezegun):
        """
        After window_seconds pass, old requests should no longer count against
        the limit, allowing new requests through.
        """
        # TODO: implement — use freezegun or fakeredis time manipulation
        ...


# ---------------------------------------------------------------------------
# Key isolation
# ---------------------------------------------------------------------------

class TestKeyIsolation:
    """Independent counters for different rate-limit keys."""

    @pytest.mark.asyncio
    async def test_different_keys_are_independent(self, rate_limiter, window_config):
        """
        Exhausting the limit for key A should not affect key B.
        """
        # TODO: implement
        ...


# ---------------------------------------------------------------------------
# Peek
# ---------------------------------------------------------------------------

class TestPeek:
    """Tests for the non-destructive peek() method."""

    @pytest.mark.asyncio
    async def test_peek_returns_correct_remaining(self, rate_limiter, window_config):
        """
        After 3 requests, peek() should report remaining = max_requests - 3.
        """
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_peek_does_not_consume_quota(self, rate_limiter, window_config):
        """
        Calling peek() N times should not affect the available quota.
        """
        # TODO: implement
        ...


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    """Tests for the reset() method."""

    @pytest.mark.asyncio
    async def test_reset_clears_window(self, rate_limiter, window_config):
        """
        After exhausting the limit and calling reset(), new requests should
        be allowed again.
        """
        # TODO: implement
        ...


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestConcurrency:
    """Atomicity tests — verify no race conditions under concurrent load."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_do_not_exceed_limit(
        self, rate_limiter, window_config
    ):
        """
        Launch max_requests + 5 concurrent check() calls via asyncio.gather.
        Exactly max_requests should succeed and exactly 5 should raise
        RateLimitExceeded.
        """
        # TODO: implement
        ...
