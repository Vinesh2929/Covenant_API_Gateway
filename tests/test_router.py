"""
tests/test_router.py — tests for the multi-provider router.
"""

from __future__ import annotations

import pytest

from gateway.router import ProviderRouter, RoutingPolicy, ModelAlias, RoutingDecision


class TestAliasResolution:

    def test_openai_model_resolves_to_openai(self, router):
        decision = router.resolve({"model": "gpt-4o-mini"})
        assert decision.provider_name == "openai"
        assert decision.canonical_model == "gpt-4o-mini"

    def test_anthropic_model_resolves_to_anthropic(self, router):
        decision = router.resolve({"model": "claude-3-5-sonnet"})
        assert decision.provider_name == "anthropic"
        assert "claude" in decision.canonical_model.lower()

    def test_local_model_resolves_to_local(self, router):
        decision = router.resolve({"model": "llama3"})
        assert decision.provider_name == "local"

    def test_unknown_model_falls_back_to_default_provider(self, router):
        decision = router.resolve({"model": "nonexistent-model-xyz"})
        assert isinstance(decision, RoutingDecision)
        assert decision.canonical_model == "nonexistent-model-xyz"


class TestRoutingPolicies:

    def test_cost_policy_selects_cheapest(self, settings):
        router = ProviderRouter(settings)
        router._policy = RoutingPolicy.COST

        cheap = ModelAlias("test-model", "openai", "cheap", cost_per_1k_tok=0.001)
        expensive = ModelAlias("test-model", "anthropic", "expensive", cost_per_1k_tok=0.1)

        selected = router._apply_policy([cheap, expensive], RoutingPolicy.COST)
        assert selected.canonical_model == "cheap"

    def test_latency_policy_selects_fastest(self, settings):
        router = ProviderRouter(settings)

        slow = ModelAlias("m", "openai", "slow", avg_latency_ms=500.0)
        fast = ModelAlias("m", "anthropic", "fast", avg_latency_ms=50.0)

        selected = router._apply_policy([slow, fast], RoutingPolicy.LATENCY)
        assert selected.canonical_model == "fast"

    def test_latency_policy_deprioritises_unmeasured(self, settings):
        router = ProviderRouter(settings)

        measured = ModelAlias("m", "openai", "measured", avg_latency_ms=200.0)
        unmeasured = ModelAlias("m", "anthropic", "unmeasured", avg_latency_ms=0.0)

        selected = router._apply_policy([measured, unmeasured], RoutingPolicy.LATENCY)
        assert selected.canonical_model == "measured"


class TestExplicitProvider:

    def test_x_provider_header_overrides_policy(self, router):
        decision = router.resolve({
            "model": "gpt-4o-mini",
            "x_provider": "openai",
        })
        assert decision.provider_name == "openai"


class TestFallback:

    def test_fallback_returns_alternative(self, settings):
        router = ProviderRouter(settings)
        router.register_alias(
            ModelAlias("multi", "openai", "multi-oai", priority=0)
        )
        router.register_alias(
            ModelAlias("multi", "anthropic", "multi-ant", priority=1)
        )

        alt = router._fallback("openai", "multi")
        assert alt is not None
        assert alt.provider_name == "anthropic"

    def test_fallback_returns_none_when_no_alternative(self, router):
        alt = router._fallback("openai", "gpt-4o-mini")
        assert alt is None


class TestRegisterAlias:

    def test_register_and_resolve(self, router):
        router.register_alias(
            ModelAlias("my-custom-model", "openai", "gpt-4o", cost_per_1k_tok=0.005)
        )
        decision = router.resolve({"model": "my-custom-model"})
        assert decision.provider_name == "openai"
        assert decision.canonical_model == "gpt-4o"


class TestListModels:

    def test_list_models_returns_entries(self, router):
        models = router.list_models()
        assert len(models) > 0
        assert all("id" in m and "owned_by" in m for m in models)

    def test_list_models_contains_gpt4o(self, router):
        models = router.list_models()
        ids = [m["id"] for m in models]
        assert "gpt-4o" in ids


class TestLatencyUpdate:

    def test_update_latency_first_measurement(self, router):
        router.update_latency("openai", "gpt-4o-mini", 100.0)
        candidates = router._aliases["gpt-4o-mini"]
        oai = [c for c in candidates if c.provider_name == "openai"][0]
        assert oai.avg_latency_ms == 100.0

    def test_update_latency_ema(self, router):
        router.update_latency("openai", "gpt-4o-mini", 100.0)
        router.update_latency("openai", "gpt-4o-mini", 200.0)
        candidates = router._aliases["gpt-4o-mini"]
        oai = [c for c in candidates if c.provider_name == "openai"][0]
        # EMA: 0.2 * 200 + 0.8 * 100 = 120
        assert abs(oai.avg_latency_ms - 120.0) < 0.01
