"""
tests/test_rate_limiter.py — sliding-window rate limiter tests.

Uses fakeredis via the conftest rate_limiter fixture.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.rate_limiter import RateLimitExceeded, WindowConfig


@pytest.fixture
def window_config() -> WindowConfig:
    return WindowConfig(max_requests=5, window_seconds=10, key_prefix="test")


class TestBasicLimiting:

    @pytest.mark.asyncio
    async def test_requests_within_limit_are_allowed(self, rate_limiter, window_config):
        for _ in range(window_config.max_requests):
            await rate_limiter.check(key="user-a", config=window_config)

    @pytest.mark.asyncio
    async def test_request_exceeding_limit_raises(self, rate_limiter, window_config):
        for _ in range(window_config.max_requests):
            await rate_limiter.check(key="user-b", config=window_config)

        with pytest.raises(RateLimitExceeded):
            await rate_limiter.check(key="user-b", config=window_config)

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_has_retry_after(self, rate_limiter, window_config):
        for _ in range(window_config.max_requests):
            await rate_limiter.check(key="user-c", config=window_config)

        with pytest.raises(RateLimitExceeded) as exc_info:
            await rate_limiter.check(key="user-c", config=window_config)

        assert exc_info.value.retry_after >= 0
        assert exc_info.value.limit == window_config.max_requests
        assert exc_info.value.window == window_config.window_seconds


class TestKeyIsolation:

    @pytest.mark.asyncio
    async def test_different_keys_are_independent(self, rate_limiter, window_config):
        for _ in range(window_config.max_requests):
            await rate_limiter.check(key="key-x", config=window_config)

        # key-x is exhausted, but key-y should still work
        await rate_limiter.check(key="key-y", config=window_config)


class TestPeek:

    @pytest.mark.asyncio
    async def test_peek_returns_correct_remaining(self, rate_limiter, window_config):
        for _ in range(3):
            await rate_limiter.check(key="peek-user", config=window_config)

        used, remaining, _ = await rate_limiter.peek(key="peek-user", config=window_config)
        assert used == 3
        assert remaining == window_config.max_requests - 3

    @pytest.mark.asyncio
    async def test_peek_does_not_consume_quota(self, rate_limiter, window_config):
        await rate_limiter.check(key="peek-nc", config=window_config)

        for _ in range(10):
            await rate_limiter.peek(key="peek-nc", config=window_config)

        used, remaining, _ = await rate_limiter.peek(key="peek-nc", config=window_config)
        assert used == 1
        assert remaining == window_config.max_requests - 1


class TestReset:

    @pytest.mark.asyncio
    async def test_reset_clears_window(self, rate_limiter, window_config):
        for _ in range(window_config.max_requests):
            await rate_limiter.check(key="reset-user", config=window_config)

        with pytest.raises(RateLimitExceeded):
            await rate_limiter.check(key="reset-user", config=window_config)

        await rate_limiter.reset(key="reset-user", config=window_config)

        # Should be allowed again after reset
        await rate_limiter.check(key="reset-user", config=window_config)


class TestConcurrency:

    @pytest.mark.asyncio
    async def test_concurrent_requests_do_not_exceed_limit(self, rate_limiter, window_config):
        total = window_config.max_requests + 5

        async def attempt():
            try:
                await rate_limiter.check(key="concurrent", config=window_config)
                return True
            except RateLimitExceeded:
                return False

        results = await asyncio.gather(*[attempt() for _ in range(total)])
        successes = sum(1 for r in results if r)
        failures = sum(1 for r in results if not r)

        assert successes == window_config.max_requests
        assert failures == 5
