"""
tests/test_router.py

Unit tests for gateway/router.py — provider routing and model alias resolution.

Test coverage goals:
  - Known model aliases resolve to the correct (provider, canonical_model) pair.
  - Unknown aliases raise ValueError or fall back to the default provider.
  - COST policy selects the cheapest alias when multiple providers match.
  - LATENCY policy selects the lowest-latency alias.
  - EXPLICIT policy (X-Provider header) overrides automatic selection.
  - Fallback logic skips a failed provider and routes to the next candidate.
  - register_alias() and list_models() behave correctly.
  - update_latency() affects subsequent LATENCY routing decisions.
"""

from __future__ import annotations

import pytest

from gateway.router import ModelAlias, ProviderRouter, RoutingPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def router(settings):
    """Return a ProviderRouter initialised with test settings."""
    # TODO: return ProviderRouter(settings)
    ...


@pytest.fixture
def gpt4_alias() -> ModelAlias:
    """Return a sample ModelAlias for gpt-4o pointing at OpenAI."""
    return ModelAlias(
        alias="gpt-4o",
        provider_name="openai",
        canonical_model="gpt-4o-2024-11-20",
        priority=1,
        cost_per_1k_tok=0.0025,
        avg_latency_ms=350.0,
    )


@pytest.fixture
def claude_alias() -> ModelAlias:
    """Return a sample ModelAlias for claude-3-5-sonnet pointing at Anthropic."""
    return ModelAlias(
        alias="claude-3-5-sonnet",
        provider_name="anthropic",
        canonical_model="claude-3-5-sonnet-20241022",
        priority=2,
        cost_per_1k_tok=0.003,
        avg_latency_ms=420.0,
    )


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------

class TestAliasResolution:
    """Tests for ProviderRouter.resolve()."""

    def test_known_alias_resolves_to_correct_provider(self, router, gpt4_alias):
        """
        Given a request with model="gpt-4o", resolve() should return a
        RoutingDecision with provider_name="openai".
        """
        # TODO: implement
        ...

    def test_unknown_alias_raises_value_error(self, router):
        """
        Given a model alias that is not in the registry and no default provider,
        resolve() should raise ValueError.
        """
        # TODO: implement
        ...

    def test_unknown_alias_falls_back_to_default_provider(self, router, settings):
        """
        When settings.providers.default_provider is set, an unknown alias should
        route to the default provider instead of raising an error.
        """
        # TODO: implement
        ...

    def test_resolve_returns_correct_adapter_class(self, router, gpt4_alias):
        """
        The RoutingDecision.adapter_class should be OpenAIAdapter for OpenAI aliases.
        """
        # TODO: implement
        ...


# ---------------------------------------------------------------------------
# Routing policies
# ---------------------------------------------------------------------------

class TestRoutingPolicies:
    """Tests for the COST, LATENCY, and EXPLICIT routing policies."""

    def test_cost_policy_selects_cheaper_provider(self, router):
        """
        Register two aliases for the same model string with different costs.
        COST policy should select the one with lower cost_per_1k_tok.
        """
        # TODO: implement
        ...

    def test_latency_policy_selects_faster_provider(self, router):
        """
        Register two aliases with different avg_latency_ms.
        LATENCY policy should select the lower-latency one.
        """
        # TODO: implement
        ...

    def test_explicit_policy_respects_x_provider_header(self, router):
        """
        When X-Provider: anthropic is in the request, resolve() should use
        Anthropic regardless of the cost/latency policy.
        """
        # TODO: implement
        ...


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

class TestFallback:
    """Tests for the provider fallback mechanism."""

    def test_fallback_skips_unavailable_provider(self, router):
        """
        Mark primary provider as unavailable; verify resolve() returns an
        alternative provider.
        """
        # TODO: implement
        ...

    def test_no_fallback_raises_when_all_providers_unavailable(self, router):
        """
        When no providers are available for an alias, resolve() should raise.
        """
        # TODO: implement
        ...


# ---------------------------------------------------------------------------
# Registry management
# ---------------------------------------------------------------------------

class TestRegistryManagement:
    """Tests for register_alias() and list_models()."""

    def test_register_alias_adds_to_registry(self, router, gpt4_alias):
        """register_alias() should make the alias resolvable."""
        # TODO: implement
        ...

    def test_register_alias_overwrites_existing(self, router, gpt4_alias):
        """Registering an alias with the same alias string should replace it."""
        # TODO: implement
        ...

    def test_list_models_returns_all_aliases(self, router, gpt4_alias, claude_alias):
        """list_models() should include every registered alias."""
        # TODO: implement
        ...

    def test_update_latency_affects_routing(self, router):
        """
        After update_latency() raises the latency of one provider, the LATENCY
        policy should prefer the other.
        """
        # TODO: implement
        ...
