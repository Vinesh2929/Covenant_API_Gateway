"""
gateway/rate_limiter.py

Distributed sliding-window rate limiter backed by Redis.

WHY A SLIDING WINDOW INSTEAD OF A FIXED WINDOW?
  A fixed window (e.g. "100 requests per minute, reset at :00 and :60") has a
  known flaw: a client can make 100 requests at :59 and 100 more at :01, for
  200 requests in 2 seconds — double the intended rate.

  A sliding window tracks the actual timestamps of recent requests.  At any
  given moment, we count only the requests that happened within the last N
  seconds from NOW, not from an arbitrary clock boundary.

WHY REDIS SORTED SETS?
  We store one entry per request in a Redis sorted set (ZSET):
    - The "member" is a unique request UUID (ensures no two requests collide)
    - The "score" is the Unix timestamp in milliseconds

  This gives us O(log N) insert and O(log N) range delete.  We can efficiently
  ask: "how many requests happened in the last 60 seconds?" by counting members
  whose score falls in the range [now-60000, now].

WHY A LUA SCRIPT?
  The sliding window algorithm requires three steps:
    1. Remove stale entries (ZREMRANGEBYSCORE)
    2. Count current entries (ZCARD)
    3. Add the new entry (ZADD) if within limit

  If we ran these as separate Redis commands, a race condition exists:
    - Thread A counts 99 (under limit)
    - Thread B counts 99 (under limit)
    - Thread A adds its request (100 total — OK)
    - Thread B adds its request (101 total — OVER LIMIT, but it's already in)

  Redis executes Lua scripts atomically — no other command runs between the
  steps.  This eliminates the race condition entirely, even across multiple
  gateway replicas hitting the same Redis instance.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional

import redis.asyncio as aioredis

from gateway.config import RedisSettings


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RateLimitExceeded(Exception):
    """
    Raised by RateLimiter.check() when the caller has exhausted their quota.

    We raise an exception (rather than returning a boolean) because the caller
    almost always wants to abort the request immediately — exceptions naturally
    unwind the call stack without requiring the caller to write:
        if not await limiter.check(...): return 429_response

    Attributes:
        key:          The rate-limit key that was exceeded.
        retry_after:  Seconds until the window slides enough to allow a new
                      request.  Suitable for the HTTP Retry-After header.
        limit:        The configured max requests per window.
        window:       The window size in seconds.
    """
    def __init__(self, key: str, retry_after: float, limit: int, window: int) -> None:
        self.key = key
        self.retry_after = retry_after
        self.limit = limit
        self.window = window
        super().__init__(
            f"Rate limit exceeded for '{key}': retry after {retry_after:.1f}s "
            f"(limit: {limit} req/{window}s)"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WindowConfig:
    """
    Parameters for a single rate-limit rule.

    Having this as a separate dataclass (rather than hardcoding values) means
    you can apply different limits to different situations:
        WindowConfig(max_requests=100, window_seconds=60)    # default per-minute
        WindowConfig(max_requests=10,  window_seconds=60)    # stricter endpoint
        WindowConfig(max_requests=1000,window_seconds=60)    # premium tier

    Attributes:
        max_requests:   Maximum requests allowed within the window.
        window_seconds: How many seconds the sliding window spans.
        key_prefix:     Namespace prefix for the Redis key.
                        "rl:apikey" → final key is "rl:apikey:<api_key>"
                        "rl:ip"     → final key is "rl:ip:<ip_address>"
    """
    max_requests: int
    window_seconds: int
    key_prefix: str = "rl"


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Async sliding-window rate limiter backed by Redis sorted sets.

    Thread / concurrency safety:
      The Lua script runs atomically on Redis.  Multiple asyncio coroutines
      (and even multiple gateway replicas) can call check() simultaneously
      without racing — Redis serialises Lua script execution server-side.

    Usage::

        limiter = RateLimiter(settings.redis)

        try:
            await limiter.check(
                key="api-key-abc123",
                config=WindowConfig(max_requests=100, window_seconds=60),
            )
        except RateLimitExceeded as exc:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(int(exc.retry_after))}
            )
    """

    # -----------------------------------------------------------------------
    # Lua script — runs atomically inside Redis.
    #
    # Arguments:
    #   KEYS[1]  — Redis sorted set key  (e.g. "rl:apikey:abc123")
    #   ARGV[1]  — current timestamp in milliseconds (integer string)
    #   ARGV[2]  — window size in milliseconds (integer string)
    #   ARGV[3]  — max requests allowed (integer string)
    #   ARGV[4]  — unique ID for this request (UUID string → ZSET member)
    #   ARGV[5]  — TTL for the sorted set key in seconds (integer string)
    #
    # Returns a 2-element Lua table (becomes a Python list):
    #   {count, 0}          request allowed; count is total entries after insert
    #   {-1, oldest_score}  limit exceeded; oldest_score is the ms timestamp
    #                       of the oldest entry (used to compute retry_after)
    # -----------------------------------------------------------------------
    _LUA_SCRIPT: str = """
local key       = KEYS[1]
local now_ms    = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local max_req   = tonumber(ARGV[3])
local req_id    = ARGV[4]
local ttl_sec   = tonumber(ARGV[5])

-- Step 1: Evict entries older than (now - window).
-- After this, only requests within the current sliding window remain.
local cutoff = now_ms - window_ms
redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)

-- Step 2: Count how many requests are in the window right now.
local count = redis.call('ZCARD', key)

-- Step 3: Reject if already at the limit.
if count >= max_req then
    -- Return the score (timestamp) of the oldest entry still in the window.
    -- Caller uses: retry_after = (oldest_ms + window_ms - now_ms) / 1000
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    if oldest and #oldest >= 2 then
        return {-1, tonumber(oldest[2])}
    end
    return {-1, now_ms}
end

-- Step 4: Record this request.
-- Score = current timestamp, Member = unique UUID.
-- The UUID as member ensures no two requests overwrite each other even if
-- they arrive at the exact same millisecond.
redis.call('ZADD', key, now_ms, req_id)

-- Step 5: Set the key TTL so Redis auto-cleans idle keys.
redis.call('EXPIRE', key, ttl_sec)

return {count + 1, 0}
"""

    def __init__(self, settings: RedisSettings) -> None:
        """
        Store config.  The Redis connection pool is created lazily on first use.

        We don't connect here to keep object construction fast and to avoid
        blocking the import.  The pool is created when the first command runs
        inside _get_client().

        Args:
            settings: The RedisSettings slice from the root Settings object.
        """
        self._settings = settings
        self._pool: Optional[aioredis.ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None

        # After the first SCRIPT LOAD call, we store the 40-char SHA-1 hash.
        # EVALSHA (hash only) is faster than EVAL (full script text) for every
        # subsequent rate-limit check.
        self._script_sha: Optional[str] = None

    async def _get_client(self) -> aioredis.Redis:
        """
        Return a Redis client, creating the connection pool on first call.

        Also loads the Lua script into Redis (SCRIPT LOAD) on first call so
        subsequent calls can use the faster EVALSHA instead of EVAL.

        Returns:
            A configured aioredis.Redis client.
        """
        if self._client is None:
            # Build the Redis URL.  The password is handled via the pool's
            # from_url() parser so it never appears in log output.
            if self._settings.password:
                url = (
                    f"redis://:{self._settings.password}"
                    f"@{self._settings.host}:{self._settings.port}"
                )
            else:
                url = f"redis://{self._settings.host}:{self._settings.port}"

            self._pool = aioredis.ConnectionPool.from_url(
                url,
                db=self._settings.db,
                max_connections=self._settings.max_connections,
                decode_responses=True,   # return str, not bytes
            )
            self._client = aioredis.Redis(connection_pool=self._pool)

        # Load the Lua script once and cache its SHA.
        # If the Redis server is restarted, its script cache is flushed.
        # In that case EVALSHA will raise a NOSCRIPT error on the next call.
        # A production-grade implementation would catch NOSCRIPT and reload.
        if self._script_sha is None:
            self._script_sha = await self._client.script_load(self._LUA_SCRIPT)

        return self._client

    async def check(self, key: str, config: WindowConfig) -> None:
        """
        Assert the caller is within their rate limit, consuming one request slot.

        Atomically:
          1. Evicts stale entries older than the window.
          2. Counts entries in the current window.
          3a. If within limit → adds this request, returns normally.
          3b. If over limit  → raises RateLimitExceeded (slot NOT consumed).

        Args:
            key:    Unique identifier for the caller (API key, IP, etc.).
                    The Redis key becomes: "{config.key_prefix}:{key}"
            config: WindowConfig defining the limit and window.

        Raises:
            RateLimitExceeded: If the limit is already exhausted.
            redis.RedisError:  If the Redis connection fails (propagates up
                               to main.py which returns a 503).
        """
        client = await self._get_client()

        # Wall-clock time in milliseconds.
        # We use time.time() (not time.monotonic()) because all gateway replicas
        # must share the same notion of "now" relative to Redis timestamps.
        now_ms = int(time.time() * 1000)
        window_ms = config.window_seconds * 1000
        redis_key = f"{config.key_prefix}:{key}"
        request_id = str(uuid.uuid4())  # unique per request, used as ZSET member
        ttl_seconds = config.window_seconds + 5   # auto-expire slightly after window

        # Execute the Lua script atomically.
        # EVALSHA sends only the 40-char SHA — faster than re-sending the full script.
        result = await client.evalsha(
            self._script_sha,
            1,                          # number of KEYS
            redis_key,                  # KEYS[1]
            str(now_ms),                # ARGV[1]
            str(window_ms),             # ARGV[2]
            str(config.max_requests),   # ARGV[3]
            request_id,                 # ARGV[4]
            str(ttl_seconds),           # ARGV[5]
        )

        count, oldest_score = int(result[0]), float(result[1])

        if count == -1:
            # Lua returned -1 → limit exceeded.
            # Calculate retry_after: how long until the oldest request exits the window.
            # Formula: oldest_entry_time + window_size = time when window slides past it.
            oldest_ms = oldest_score
            retry_after_ms = (oldest_ms + window_ms) - now_ms
            retry_after = max(0.0, retry_after_ms / 1000)

            raise RateLimitExceeded(
                key=key,
                retry_after=retry_after,
                limit=config.max_requests,
                window=config.window_seconds,
            )
        # count > 0 → request allowed, proceed normally.

    async def peek(
        self, key: str, config: WindowConfig
    ) -> tuple[int, int, float]:
        """
        Return current usage stats without consuming a request slot.

        Unlike check(), this does NOT add an entry to the sorted set.
        Useful for the /health endpoint, admin APIs, and tests.

        Implementation:
          We run ZREMRANGEBYSCORE + ZCARD + ZRANGE in a single Redis pipeline
          (one round-trip instead of three) to get a consistent snapshot.

        Args:
            key:    Caller identifier.
            config: WindowConfig for this limit tier.

        Returns:
            A 3-tuple:
              - requests_used:      Number of requests in the current window.
              - requests_remaining: How many more are allowed.
              - reset_at:           Unix timestamp (float seconds) when the
                                    oldest entry in the window will expire.
        """
        client = await self._get_client()

        now_ms = int(time.time() * 1000)
        window_ms = config.window_seconds * 1000
        cutoff = now_ms - window_ms
        redis_key = f"{config.key_prefix}:{key}"

        # Pipeline batches all three commands into one network round-trip.
        # transaction=True wraps them in MULTI/EXEC for consistency.
        async with client.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(redis_key, "-inf", cutoff)   # clean stale entries
            pipe.zcard(redis_key)                               # count in-window entries
            pipe.zrange(redis_key, 0, 0, withscores=True)      # oldest entry
            results = await pipe.execute()

        # results[0] = zremrangebyscore result (number of removed items)
        # results[1] = zcard result (current count)
        # results[2] = zrange result (list of (member, score) tuples)
        count: int = results[1]
        oldest_entries: list = results[2]

        remaining = max(0, config.max_requests - count)

        if oldest_entries:
            # oldest_entries is [(member, score)] — we want the score (timestamp in ms)
            oldest_ms = oldest_entries[0][1]
            reset_at = (oldest_ms + window_ms) / 1000  # convert ms → Unix seconds
        else:
            # Empty window — no requests recorded, quota fully available.
            reset_at = time.time()

        return count, remaining, reset_at

    async def reset(self, key: str, config: WindowConfig) -> None:
        """
        Clear the rate-limit window for a key by deleting its Redis sorted set.

        Intended for:
          - Test teardown (ensure clean state between test cases)
          - Admin API (manually unblock a caller who hit the limit)

        Args:
            key:    Caller identifier.
            config: WindowConfig that determines the Redis key name.
        """
        client = await self._get_client()
        redis_key = f"{config.key_prefix}:{key}"
        await client.delete(redis_key)

    async def close(self) -> None:
        """
        Gracefully shut down the Redis connection pool.

        Called from the FastAPI shutdown lifespan hook.  Without this, the
        Redis server logs "connection reset by peer" errors as the process exits.

        After close(), this instance should not be used.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._pool is not None:
            await self._pool.disconnect()
            self._pool = None
