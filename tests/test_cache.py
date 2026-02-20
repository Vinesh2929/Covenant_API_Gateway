"""
tests/test_cache.py — tests for the semantic cache.

Uses the StubEmbedder and in-memory FAISS from conftest.
"""

from __future__ import annotations

import pytest


def _make_response(content: str = "cached answer") -> dict:
    return {
        "id": "chatcmpl-cached",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class TestCacheMissAndHit:

    @pytest.mark.asyncio
    async def test_empty_cache_returns_none(self, semantic_cache):
        result = await semantic_cache.get("What is Python?")
        assert result is None

    @pytest.mark.asyncio
    async def test_store_then_get_returns_hit(self, semantic_cache):
        prompt = "What is Python?"
        response = _make_response("Python is a programming language.")

        await semantic_cache.set(prompt, response)
        hit = await semantic_cache.get(prompt)

        assert hit is not None
        assert hit.payload["choices"][0]["message"]["content"] == "Python is a programming language."
        assert hit.similarity >= 0.92

    @pytest.mark.asyncio
    async def test_different_prompt_returns_miss(self, semantic_cache):
        await semantic_cache.set(
            "Explain quantum computing", _make_response("Quantum computers use qubits"),
        )
        result = await semantic_cache.get("How do I bake a cake?")
        assert result is None


class TestCacheInvalidation:

    @pytest.mark.asyncio
    async def test_invalidate_removes_entry(self, semantic_cache):
        prompt = "Hello there"
        response = _make_response("General Kenobi")

        cache_key = await semantic_cache.set(prompt, response)
        hit = await semantic_cache.get(prompt)
        assert hit is not None

        await semantic_cache.invalidate(cache_key)

        miss = await semantic_cache.get(prompt)
        assert miss is None


class TestCacheStats:

    @pytest.mark.asyncio
    async def test_stats_track_hits_and_misses(self, semantic_cache):
        prompt = "Weather in London"
        response = _make_response("It rains a lot")

        # Miss
        await semantic_cache.get(prompt)

        # Store
        await semantic_cache.set(prompt, response)

        # Hit
        await semantic_cache.get(prompt)

        s = semantic_cache.stats
        assert s["hits"] >= 1
        assert s["misses"] >= 1
        assert 0.0 <= s["hit_rate"] <= 1.0


class TestCacheCapacity:

    @pytest.mark.asyncio
    async def test_multiple_entries_coexist(self, semantic_cache):
        for i in range(5):
            await semantic_cache.set(
                f"Unique question number {i} about topic {i}",
                _make_response(f"Answer number {i}"),
            )

        for i in range(5):
            result = await semantic_cache.get(f"Unique question number {i} about topic {i}")
            assert result is not None
            assert f"Answer number {i}" in result.payload["choices"][0]["message"]["content"]
