"""
tests/conftest.py — shared pytest fixtures for the entire test suite.

Uses fakeredis (in-memory) and mocks so no external services are needed.
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
import fakeredis.aioredis

from gateway.config import (
    CacheSettings,
    ContractSettings,
    LangfuseSettings,
    ProviderSettings,
    RedisSettings,
    SecuritySettings,
    Settings,
)
from gateway.rate_limiter import RateLimiter
from gateway.security.pattern_guard import PatternGuard
from gateway.security.ml_guard import MLGuard, ScanResult
from gateway.security.guard import SecurityGuard
from gateway.cache.store import FAISSStore
from gateway.cache.semantic_cache import SemanticCache
from gateway.contracts.registry import ContractRegistry
from gateway.router import ProviderRouter


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def settings():
    return Settings(
        app_name="Test Gateway",
        app_version="0.0.1-test",
        environment="test",
        gateway_api_key="test-key-12345678",
        redis=RedisSettings(host="localhost", port=6379, password=None),
        langfuse=LangfuseSettings(enabled=False),
        providers=ProviderSettings(
            openai_api_key="sk-test-fake",
            anthropic_api_key="sk-ant-test-fake",
        ),
        security=SecuritySettings(
            pattern_file_path="gateway/security/patterns.json",
            pattern_guard_enabled=True,
            ml_guard_enabled=True,
            ml_confidence_threshold=0.85,
        ),
        cache=CacheSettings(
            enabled=True,
            similarity_threshold=0.92,
            max_cache_entries=100,
            cache_ttl_seconds=60,
        ),
        contracts=ContractSettings(
            contracts_dir="contracts/",
            drift_enabled=False,
        ),
    )


# ---------------------------------------------------------------------------
# Redis (fakeredis — fully in-memory)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def redis_client():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def rate_limiter(redis_client):
    limiter = RateLimiter.__new__(RateLimiter)
    limiter._settings = RedisSettings()
    limiter._pool = None
    limiter._client = redis_client
    limiter._script_sha = await redis_client.script_load(RateLimiter._LUA_SCRIPT)
    return limiter


# ---------------------------------------------------------------------------
# Security guards
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pattern_guard():
    return PatternGuard("gateway/security/patterns.json")


@pytest.fixture
def ml_guard_stub(settings):
    guard = MLGuard.__new__(MLGuard)
    guard._settings = settings.security
    guard._model = MagicMock()
    guard._tokenizer = MagicMock()
    guard._device = "cpu"
    guard._loaded = True

    async def _stub_scan(text: str) -> ScanResult:
        import time
        if "injection" in text.lower() or "ignore previous" in text.lower():
            return ScanResult(
                is_injection=True,
                confidence=0.95,
                label="INJECTION",
                latency_ms=1.0,
                model_version="stub",
            )
        return ScanResult(
            is_injection=False,
            confidence=0.05,
            label="SAFE",
            latency_ms=1.0,
            model_version="stub",
        )

    guard.scan = _stub_scan
    return guard


@pytest.fixture
def security_guard(settings, pattern_guard, ml_guard_stub):
    guard = SecurityGuard.__new__(SecurityGuard)
    guard._settings = settings.security
    guard._pattern_guard = pattern_guard
    guard._ml_guard = ml_guard_stub
    guard._dry_run = False
    return guard


# ---------------------------------------------------------------------------
# Cache (stub embedder + in-memory FAISS)
# ---------------------------------------------------------------------------

EMBED_DIM = 32  # 32-dim vectors: low chance of spurious similarity (std dev ~0.18)


class StubEmbedder:
    """Deterministic embedder for testing — hashes text into 32-dim unit vectors.

    Uses SHA256 as a seed for numpy's RNG so that:
    - Same text always produces the same vector (fully deterministic).
    - Different texts produce well-separated vectors (cosine sim ~ 0 expected,
      std dev ~0.18 in 32D — probability of spurious > 0.92 hit is < 0.001%).
    """

    def __init__(self):
        self._dimension = EMBED_DIM

    @property
    def dimension(self):
        return self._dimension

    async def warm_up(self):
        pass

    async def embed(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        # Use the full 256-bit hash as a seed for a reproducible RNG.
        # numpy default_rng accepts arbitrary integers as seed.
        seed = int.from_bytes(h, "little") % (2**31)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(EMBED_DIM).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        vecs = [await self.embed(t) for t in texts]
        return np.stack(vecs)


@pytest.fixture
def embedder_stub():
    return StubEmbedder()


@pytest.fixture
def faiss_store(settings):
    store = FAISSStore(settings.cache)
    store.build(EMBED_DIM)
    return store


@pytest_asyncio.fixture
async def semantic_cache(settings, embedder_stub, faiss_store, redis_client):
    cache = SemanticCache.__new__(SemanticCache)
    cache._settings = settings.cache
    cache._redis = redis_client
    cache._embedder = embedder_stub
    cache._store = faiss_store
    cache._hits = 0
    cache._misses = 0
    return cache


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

@pytest.fixture
def contract_registry():
    registry = ContractRegistry(contracts_dir="contracts/")
    registry.load()
    return registry


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

@pytest.fixture
def router(settings):
    return ProviderRouter(settings)


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_request() -> dict:
    return {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "temperature": 0.7,
        "max_tokens": 256,
    }


@pytest.fixture
def sample_response() -> dict:
    return {
        "id": "chatcmpl-test-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "The answer is 4."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
        },
    }
