"""
tests/conftest.py

Shared pytest fixtures for the entire test suite.

Fixtures defined here are automatically available to all test modules without
explicit imports.  Use the narrowest possible scope (function > module > session)
to keep tests isolated and side-effect free.

Fixtures provided:
  - settings            — a Settings instance with test-safe defaults
  - redis_client        — a fakeredis.FakeAsyncRedis instance (no real Redis needed)
  - rate_limiter        — a RateLimiter wired to the fake Redis
  - pattern_guard       — a PatternGuard loaded from a minimal test patterns file
  - ml_guard_stub       — an MLGuard with the model replaced by a stub that returns
                          a configurable ScanResult
  - security_guard      — a SecurityGuard using pattern_guard + ml_guard_stub
  - embedder_stub       — an Embedder whose embed() returns deterministic vectors
  - faiss_store         — a fresh in-memory FAISSStore (dimension=8 for speed)
  - semantic_cache      — a SemanticCache using embedder_stub + faiss_store + fake Redis
  - contract_registry   — a ContractRegistry loaded from tests/fixtures/contracts/
  - test_client         — FastAPI TestClient wrapping the full app with mocked deps
  - sample_request      — a minimal valid chat completion request dict
  - sample_response     — a minimal valid normalised response dict
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def settings():
    """
    Return a Settings instance configured for testing.

    Uses safe defaults: no real API keys, ML guard disabled, Langfuse disabled,
    a small rate limit window for fast tests.
    """
    # TODO: implement — return Settings(...)
    ...


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------

@pytest.fixture
async def redis_client():
    """
    Return a fakeredis.FakeAsyncRedis instance.

    fakeredis implements the full aioredis interface in memory, making tests
    fast and network-free.
    """
    # TODO: implement — import fakeredis.aioredis, yield FakeAsyncRedis()
    ...


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

@pytest.fixture
async def rate_limiter(settings, redis_client):
    """
    Return a RateLimiter wired to the fake Redis client.
    """
    # TODO: implement
    ...


# ---------------------------------------------------------------------------
# Security guards
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pattern_guard(settings):
    """
    Return a PatternGuard loaded with a small set of test patterns.
    """
    # TODO: implement — PatternGuard("tests/fixtures/test_patterns.json")
    ...


@pytest.fixture
def ml_guard_stub(settings):
    """
    Return an MLGuard whose _infer() method is replaced with a configurable stub.

    Tests can override the stub's return value to simulate injection detections
    without loading the actual DistilBERT model.
    """
    # TODO: implement — patch MLGuard._infer with MagicMock
    ...


@pytest.fixture
def security_guard(settings, pattern_guard, ml_guard_stub):
    """
    Return a SecurityGuard composed from the pattern_guard fixture and the
    ml_guard_stub (no real ML model needed).
    """
    # TODO: implement
    ...


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@pytest.fixture
def embedder_stub():
    """
    Return an Embedder whose embed() returns deterministic 8-dimensional unit
    vectors derived from the hash of the input text.

    Using 8D instead of 384D keeps FAISS operations fast in tests.
    """
    # TODO: implement — mock Embedder with deterministic embed()
    ...


@pytest.fixture
def faiss_store(settings):
    """
    Return a fresh in-memory FAISSStore with dimension=8 (for speed).
    """
    # TODO: implement — FAISSStore(settings.cache), call build(8)
    ...


@pytest.fixture
async def semantic_cache(settings, embedder_stub, faiss_store, redis_client):
    """
    Return a SemanticCache with the stub embedder, in-memory FAISS, and
    fake Redis — no external services required.
    """
    # TODO: implement
    ...


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def contract_registry():
    """
    Return a ContractRegistry loaded from tests/fixtures/contracts/.

    The fixture directory contains a small set of contracts for each type
    (keyword, regex, sentiment, schema) used in test_contracts.py.
    """
    # TODO: implement — ContractRegistry("tests/fixtures/contracts/"), call load()
    ...


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_request() -> dict:
    """
    Return a minimal valid normalised chat completion request dict.
    """
    return {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "temperature": 0.7,
        "max_tokens": 256,
    }


@pytest.fixture
def sample_response() -> dict:
    """
    Return a minimal valid normalised chat completion response dict.
    """
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


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client(settings) -> TestClient:
    """
    Return a FastAPI TestClient for the full gateway app with mocked external
    dependencies injected via FastAPI's dependency_overrides.

    Covers: mocked provider adapters, fake Redis, stub ML model.
    """
    # TODO: implement — import app, override dependencies, return TestClient(app)
    ...
