"""
gateway/router.py

Provider routing logic and model alias resolution.

Responsibilities:
  - Maintain a registry that maps model alias strings (e.g. "gpt-4o", "claude-3-5-sonnet",
    "llama3") to the correct provider name and canonical model identifier.
  - Inspect each incoming request's `model` field and resolve it to a
    (provider_name, canonical_model_id) pair.
  - Support rule-based routing policies: cost optimisation (prefer cheaper
    models for short prompts), latency optimisation (prefer lowest-latency
    provider), and explicit overrides via request headers.
  - Detect unavailable providers and fall back to alternatives according to a
    configurable priority list.
  - Expose a single `resolve()` method that returns a fully-populated
    `RoutingDecision` dataclass consumed by main.py.

Key classes / functions:
  - ModelAlias              — dataclass: alias → provider + canonical_model + priority
  - RoutingDecision         — dataclass: provider_name, canonical_model, adapter_class
  - RoutingPolicy           — Enum: COST, LATENCY, EXPLICIT
  - ProviderRouter          — main class; holds the alias registry and routing logic
    - __init__(settings)    — loads alias map from config/settings
    - resolve(request_body) — returns a RoutingDecision for a given chat request
    - _apply_policy()       — internal: selects provider according to active policy
    - _fallback()           — internal: tries the next provider if primary is down
    - register_alias()      — add or update a model alias at runtime
    - list_models()         — returns all registered aliases (used by /models endpoint)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Type

from gateway.config import Settings
from gateway.providers.base import BaseProviderAdapter


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelAlias:
    """
    Mapping from a user-facing model alias to a specific provider and model.

    Attributes:
        alias:            The string a client passes in the `model` field.
        provider_name:    One of "openai", "anthropic", "local".
        canonical_model:  The exact model ID the provider expects.
        priority:         Lower numbers are preferred during fallback resolution.
        cost_per_1k_tok:  Approximate USD cost, used by the COST routing policy.
        avg_latency_ms:   Rolling average latency, used by the LATENCY policy.
    """
    alias: str
    provider_name: str
    canonical_model: str
    priority: int = 0
    cost_per_1k_tok: float = 0.0
    avg_latency_ms: float = 0.0


@dataclass
class RoutingDecision:
    """
    Output of ProviderRouter.resolve().  Contains everything main.py needs to
    forward the request to the right provider adapter.

    Attributes:
        provider_name:   Human-readable provider identifier.
        canonical_model: Model ID to send to the provider API.
        adapter_class:   The concrete BaseProviderAdapter subclass to use.
        alias_used:      The original alias string from the request.
    """
    provider_name: str
    canonical_model: str
    adapter_class: Type[BaseProviderAdapter]
    alias_used: str


class RoutingPolicy(str, Enum):
    """
    Controls which heuristic the router uses when multiple providers can serve
    a given model alias.

    COST:     Prefer the provider / model with the lowest cost_per_1k_tok.
    LATENCY:  Prefer the provider with the lowest rolling average latency.
    EXPLICIT: Use the provider explicitly requested via the X-Provider header.
    """
    COST = "cost"
    LATENCY = "latency"
    EXPLICIT = "explicit"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class ProviderRouter:
    """
    Central routing component.  Resolves model aliases and selects the
    appropriate provider adapter for each incoming request.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialise the router with the application settings.

        Builds the default alias registry from the built-in model list and
        applies any overrides defined in the settings / environment.

        Args:
            settings: The root Settings object (provides provider API keys,
                      default provider, routing policy, etc.).
        """
        # TODO: implement — populate self._aliases: dict[str, ModelAlias]
        self._aliases: dict[str, ModelAlias] = {}
        self._policy: RoutingPolicy = RoutingPolicy.COST
        self._settings = settings

    def resolve(self, request_body: dict) -> RoutingDecision:
        """
        Resolve the `model` field in a chat completion request body to a
        RoutingDecision.

        Resolution order:
          1. Check for X-Provider override header (EXPLICIT policy).
          2. Look up the alias in the internal registry.
          3. Apply the active routing policy if multiple providers match.
          4. Attempt fallback if the primary provider is marked unavailable.

        Args:
            request_body: Parsed JSON body of the incoming chat request.

        Returns:
            A RoutingDecision with the selected provider, model, and adapter.

        Raises:
            ValueError: If the model alias is not recognised and no default
                        provider is configured.
        """
        # TODO: implement
        ...

    def _apply_policy(
        self,
        candidates: list[ModelAlias],
        policy: RoutingPolicy,
    ) -> ModelAlias:
        """
        Given a list of candidate aliases that all match the requested model,
        apply the routing policy to select the best one.

        Args:
            candidates: Non-empty list of matching ModelAlias objects.
            policy:     The active RoutingPolicy enum value.

        Returns:
            The selected ModelAlias.
        """
        # TODO: implement
        ...

    def _fallback(self, failed_provider: str, alias: str) -> Optional[ModelAlias]:
        """
        Attempt to find an alternative provider for `alias` after
        `failed_provider` is deemed unavailable.

        Args:
            failed_provider: The provider name that failed.
            alias:           The original model alias requested.

        Returns:
            An alternative ModelAlias, or None if no fallback exists.
        """
        # TODO: implement
        ...

    def register_alias(self, alias: ModelAlias) -> None:
        """
        Add or replace a ModelAlias in the runtime registry.

        Useful for dynamic alias registration without restarting the gateway
        (e.g. when a new Ollama model is pulled).

        Args:
            alias: The ModelAlias dataclass to register.
        """
        # TODO: implement
        ...

    def list_models(self) -> list[dict]:
        """
        Return a list of all registered model aliases and their metadata.

        Used by the GET /v1/models endpoint to advertise available models in
        the OpenAI-compatible format.

        Returns:
            List of dicts compatible with the OpenAI /v1/models response shape.
        """
        # TODO: implement
        ...

    def update_latency(self, provider_name: str, alias: str, latency_ms: float) -> None:
        """
        Update the rolling average latency for a provider/alias pair.

        Called by main.py after each successful upstream request so the LATENCY
        routing policy has up-to-date data.

        Args:
            provider_name: The provider that served the request.
            alias:         The model alias that was used.
            latency_ms:    The measured round-trip latency in milliseconds.
        """
        # TODO: implement
        ...
