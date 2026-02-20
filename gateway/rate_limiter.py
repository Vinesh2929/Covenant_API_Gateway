"""
gateway/rate_limiter.py

Distributed sliding-window rate limiter backed by Redis.

Responsibilities:
  - Implement the sliding-window log algorithm using Redis sorted sets so that
    rate limits are enforced consistently across multiple gateway replicas.
  - Support multiple limit tiers keyed by: API key, IP address, or
    (API-key, model) pairs — giving fine-grained control.
  - Expose a simple async `check()` interface that either grants the request
    or raises an HTTP 429 with a Retry-After header value.
  - Provide a `peek()` method that returns remaining quota without consuming it
    (used by the /health endpoint and monitoring scripts).
  - Use an atomic Lua script executed on Redis to avoid race conditions between
    the "remove stale entries → count → add new entry" steps.

Key classes / functions:
  - RateLimitExceeded       — exception raised when a limit is hit
  - WindowConfig            — dataclass: max_requests, window_seconds, key_prefix
  - RateLimiter             — main class
    - __init__(settings)    — creates async Redis connection pool
    - check(key, config)    — async: enforce limit, raise RateLimitExceeded on breach
    - peek(key, config)     — async: return (used, remaining, reset_at) without side-effects
    - close()               — async: cleanly close the Redis pool on shutdown
    - _sliding_window_lua   — class-level Lua script string loaded into Redis
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import redis.asyncio as aioredis

from gateway.config import RedisSettings


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RateLimitExceeded(Exception):
    """
    Raised by RateLimiter.check() when the caller has exhausted their quota
    for the current sliding window.

    Attributes:
        key:          The rate-limit key that was exceeded (e.g. "api:abc123").
        retry_after:  Seconds until the oldest request drops out of the window,
                      suitable for the Retry-After HTTP response header.
        limit:        The configured maximum number of requests per window.
        window:       The window size in seconds.
    """
    def __init__(self, key: str, retry_after: float, limit: int, window: int) -> None:
        self.key = key
        self.retry_after = retry_after
        self.limit = limit
        self.window = window
        super().__init__(
            f"Rate limit exceeded for '{key}': retry after {retry_after:.1f}s"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WindowConfig:
    """
    Parameters for a single rate-limit rule.

    Attributes:
        max_requests:    Maximum number of requests allowed in the window.
        window_seconds:  Length of the sliding window in seconds.
        key_prefix:      Redis key namespace (e.g. "rl:ip", "rl:apikey").
    """
    max_requests: int
    window_seconds: int
    key_prefix: str = "rl"


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Async sliding-window rate limiter.

    Uses a Redis sorted set per key where each member is a unique request UUID
    and the score is the Unix timestamp (float).  An atomic Lua script handles
    the read-modify-write cycle to avoid race conditions.

    Usage::

        limiter = RateLimiter(settings.redis)
        await limiter.check(
            key="api:my-key-123",
            config=WindowConfig(max_requests=100, window_seconds=60),
        )
    """

    # Lua script executed atomically on the Redis server.
    # Arguments: KEYS[1]=sorted-set key, ARGV[1]=now_ms, ARGV[2]=window_ms,
    #            ARGV[3]=max_requests, ARGV[4]=request_id, ARGV[5]=ttl_seconds
    # Returns: {current_count, oldest_timestamp_ms} or error string.
    _LUA_SCRIPT: str = """
    -- TODO: implement sliding window Lua script
    -- 1. ZREMRANGEBYSCORE KEYS[1] 0 (ARGV[1] - ARGV[2])   -- evict stale entries
    -- 2. count = ZCARD KEYS[1]
    -- 3. if count >= tonumber(ARGV[3]) then return error
    -- 4. ZADD KEYS[1] ARGV[1] ARGV[4]
    -- 5. EXPIRE KEYS[1] ARGV[5]
    -- 6. return count + 1
    return {0, 0}
    """

    def __init__(self, settings: RedisSettings) -> None:
        """
        Create the async Redis connection pool.

        The pool is not connected until the first command is issued (lazy init).

        Args:
            settings: RedisSettings slice from the root Settings object.
        """
        # TODO: implement — create aioredis.ConnectionPool and store as self._pool
        self._settings = settings
        self._pool: Optional[aioredis.ConnectionPool] = None
        self._script_sha: Optional[str] = None  # SHA of loaded Lua script

    async def _get_client(self) -> aioredis.Redis:
        """
        Return a Redis client from the pool, initialising the pool on first call.

        Also loads the Lua script into Redis (SCRIPT LOAD) and caches its SHA
        so subsequent calls use EVALSHA for efficiency.

        Returns:
            An aioredis.Redis client instance.
        """
        # TODO: implement
        ...

    async def check(self, key: str, config: WindowConfig) -> None:
        """
        Assert that the caller identified by `key` is within their rate limit.

        Atomically records the current request in the sliding window.  If the
        limit is exceeded, raises RateLimitExceeded with the seconds until the
        caller can retry.

        Args:
            key:    Unique identifier for the caller (e.g. API key or IP).
            config: WindowConfig specifying the limit and window size.

        Raises:
            RateLimitExceeded: If max_requests have already been made within
                               the current sliding window.
        """
        # TODO: implement
        ...

    async def peek(
        self, key: str, config: WindowConfig
    ) -> tuple[int, int, float]:
        """
        Return the current usage for `key` without consuming a request slot.

        Args:
            key:    Caller identifier.
            config: WindowConfig for this limit tier.

        Returns:
            A 3-tuple: (requests_used, requests_remaining, reset_at_unix_timestamp)
        """
        # TODO: implement
        ...

    async def reset(self, key: str, config: WindowConfig) -> None:
        """
        Clear all request records for `key` in the given window config.

        Intended for testing and admin tooling only.

        Args:
            key:    Caller identifier.
            config: WindowConfig that determines the Redis key name.
        """
        # TODO: implement
        ...

    async def close(self) -> None:
        """
        Gracefully close the Redis connection pool.

        Should be called from the FastAPI shutdown lifespan hook.
        """
        # TODO: implement
        ...
