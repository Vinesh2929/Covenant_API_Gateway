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

import time
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
# Built-in alias table
# ---------------------------------------------------------------------------

def _build_default_aliases() -> list[ModelAlias]:
    """Return the built-in alias list covering OpenAI, Anthropic, and Ollama."""
    return [
        # OpenAI models
        ModelAlias("gpt-4o",            "openai", "gpt-4o",                       priority=0, cost_per_1k_tok=0.005),
        ModelAlias("gpt-4o-mini",       "openai", "gpt-4o-mini",                  priority=0, cost_per_1k_tok=0.00015),
        ModelAlias("gpt-4-turbo",       "openai", "gpt-4-turbo",                  priority=0, cost_per_1k_tok=0.01),
        ModelAlias("gpt-4",             "openai", "gpt-4",                         priority=0, cost_per_1k_tok=0.03),
        ModelAlias("gpt-3.5-turbo",     "openai", "gpt-3.5-turbo",                priority=0, cost_per_1k_tok=0.0015),

        # Anthropic Claude models
        ModelAlias("claude-3-5-sonnet", "anthropic", "claude-3-5-sonnet-20241022", priority=0, cost_per_1k_tok=0.003),
        ModelAlias("claude-3-5-haiku",  "anthropic", "claude-3-5-haiku-20241022",  priority=0, cost_per_1k_tok=0.0008),
        ModelAlias("claude-3-opus",     "anthropic", "claude-3-opus-20240229",      priority=0, cost_per_1k_tok=0.015),
        ModelAlias("claude-3-sonnet",   "anthropic", "claude-3-sonnet-20240229",    priority=0, cost_per_1k_tok=0.003),
        ModelAlias("claude-3-haiku",    "anthropic", "claude-3-haiku-20240307",     priority=0, cost_per_1k_tok=0.00025),

        # Local / Ollama models (free — cost 0)
        ModelAlias("llama3",    "local", "llama3",    priority=0, cost_per_1k_tok=0.0),
        ModelAlias("llama3.1",  "local", "llama3.1",  priority=0, cost_per_1k_tok=0.0),
        ModelAlias("llama3.2",  "local", "llama3.2",  priority=0, cost_per_1k_tok=0.0),
        ModelAlias("mistral",   "local", "mistral",   priority=0, cost_per_1k_tok=0.0),
        ModelAlias("codellama", "local", "codellama", priority=0, cost_per_1k_tok=0.0),
        ModelAlias("gemma2",    "local", "gemma2",    priority=0, cost_per_1k_tok=0.0),
        ModelAlias("phi3",      "local", "phi3",      priority=0, cost_per_1k_tok=0.0),
        ModelAlias("qwen2",     "local", "qwen2",     priority=0, cost_per_1k_tok=0.0),
    ]


def _get_adapter_class(provider_name: str) -> Type[BaseProviderAdapter]:
    """
    Return the concrete adapter class for the given provider name.

    Imported lazily (inside this function) to avoid circular imports.

    Args:
        provider_name: One of "openai", "anthropic", "local".

    Returns:
        The adapter class (not an instance).

    Raises:
        ValueError: If the provider name is not known.
    """
    if provider_name == "openai":
        from gateway.providers.openai_adapter import OpenAIAdapter
        return OpenAIAdapter
    elif provider_name == "anthropic":
        from gateway.providers.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter
    elif provider_name == "local":
        from gateway.providers.local_adapter import LocalAdapter
        return LocalAdapter
    else:
        raise ValueError(f"Unknown provider: {provider_name!r}")


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
        self._settings = settings

        # Parse the policy from settings (default to COST).
        policy_str = getattr(settings, "routing_policy", "cost")
        try:
            self._policy = RoutingPolicy(policy_str)
        except ValueError:
            self._policy = RoutingPolicy.COST

        # _aliases maps alias string → list[ModelAlias].
        # A list supports having the same alias served by multiple providers.
        self._aliases: dict[str, list[ModelAlias]] = {}

        # Populate with built-in aliases.
        for alias_obj in _build_default_aliases():
            self._aliases.setdefault(alias_obj.alias, []).append(alias_obj)

    def resolve(self, request_body: dict) -> RoutingDecision:
        """
        Resolve the `model` field in a chat completion request body to a
        RoutingDecision.

        Resolution order:
          1. Check for X-Provider override in request (injected from header).
          2. Look up the alias in the internal registry.
          3. Apply the active routing policy if multiple providers match.
          4. Fall back to the default provider if alias not registered.

        Args:
            request_body: Parsed JSON body of the incoming chat request.

        Returns:
            A RoutingDecision with the selected provider, model, and adapter.

        Raises:
            ValueError: If the model alias is not recognised and no default
                        provider is configured.
        """
        alias = request_body.get("model", "")

        # X-Provider override — set by main.py from the X-Provider header.
        explicit_provider = request_body.get("x_provider")

        candidates = list(self._aliases.get(alias, []))

        if not candidates:
            # Alias not in registry — try using it as a literal canonical model ID
            # routed to the default provider.
            default_provider = getattr(self._settings, "default_provider", "openai")
            candidates = [
                ModelAlias(
                    alias=alias,
                    provider_name=default_provider,
                    canonical_model=alias,
                    priority=0,
                )
            ]

        if explicit_provider:
            # Filter to the explicitly requested provider if any match.
            filtered = [c for c in candidates if c.provider_name == explicit_provider]
            if filtered:
                candidates = filtered

        selected = self._apply_policy(candidates, self._policy)

        return RoutingDecision(
            provider_name=selected.provider_name,
            canonical_model=selected.canonical_model,
            adapter_class=_get_adapter_class(selected.provider_name),
            alias_used=alias,
        )

    def _apply_policy(
        self,
        candidates: list[ModelAlias],
        policy: RoutingPolicy,
    ) -> ModelAlias:
        """
        Given a list of candidate aliases that all match the requested model,
        apply the routing policy to select the best one.

        COST policy:    Sort by cost_per_1k_tok ascending → pick cheapest.
        LATENCY policy: Sort by avg_latency_ms ascending → pick fastest.
        EXPLICIT:       Use priority (lowest number = highest priority).

        Args:
            candidates: Non-empty list of matching ModelAlias objects.
            policy:     The active RoutingPolicy enum value.

        Returns:
            The selected ModelAlias.
        """
        if len(candidates) == 1:
            return candidates[0]

        if policy == RoutingPolicy.COST:
            return min(candidates, key=lambda a: a.cost_per_1k_tok)
        elif policy == RoutingPolicy.LATENCY:
            # Aliases with avg_latency_ms == 0 have never been measured;
            # treat as infinity so they are deprioritised.
            return min(
                candidates,
                key=lambda a: a.avg_latency_ms if a.avg_latency_ms > 0 else float("inf"),
            )
        else:
            # EXPLICIT or unknown — return the highest-priority candidate.
            return min(candidates, key=lambda a: a.priority)

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
        candidates = self._aliases.get(alias, [])
        alternatives = [c for c in candidates if c.provider_name != failed_provider]
        if not alternatives:
            return None
        return self._apply_policy(alternatives, self._policy)

    def register_alias(self, alias: ModelAlias) -> None:
        """
        Add or replace a ModelAlias in the runtime registry.

        If an alias with the same (alias, provider_name) pair already exists,
        it is replaced.  Otherwise, the new alias is appended.

        Useful for dynamic alias registration without restarting the gateway
        (e.g. when a new Ollama model is pulled).

        Args:
            alias: The ModelAlias dataclass to register.
        """
        existing = self._aliases.setdefault(alias.alias, [])

        # Replace existing entry with the same provider, or append.
        for i, entry in enumerate(existing):
            if entry.provider_name == alias.provider_name:
                existing[i] = alias
                return
        existing.append(alias)

    def list_models(self) -> list[dict]:
        """
        Return a list of all registered model aliases and their metadata.

        Used by the GET /v1/models endpoint to advertise available models in
        the OpenAI-compatible format.

        Returns:
            List of dicts compatible with the OpenAI /v1/models response shape.
        """
        seen: set[str] = set()
        models = []

        for alias_str, alias_list in sorted(self._aliases.items()):
            for alias_obj in alias_list:
                key = f"{alias_str}:{alias_obj.provider_name}"
                if key in seen:
                    continue
                seen.add(key)

                models.append({
                    "id": alias_str,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": alias_obj.provider_name,
                    "canonical_model": alias_obj.canonical_model,
                    "cost_per_1k_tokens": alias_obj.cost_per_1k_tok,
                    "avg_latency_ms": alias_obj.avg_latency_ms,
                })

        return models

    def update_latency(self, provider_name: str, alias: str, latency_ms: float) -> None:
        """
        Update the rolling average latency for a provider/alias pair.

        Uses an exponential moving average (EMA) with alpha=0.2 to smooth
        out noisy measurements.

            new_avg = alpha * new_sample + (1 - alpha) * old_avg

        A low alpha (0.2) gives more weight to historical data — good for
        stable latencies.  Increase alpha for faster reaction to changes.

        Called by main.py after each successful upstream request so the LATENCY
        routing policy has up-to-date data.

        Args:
            provider_name: The provider that served the request.
            alias:         The model alias that was used.
            latency_ms:    The measured round-trip latency in milliseconds.
        """
        candidates = self._aliases.get(alias, [])
        alpha = 0.2   # EMA smoothing factor

        for entry in candidates:
            if entry.provider_name == provider_name:
                if entry.avg_latency_ms == 0.0:
                    # First measurement — use the sample directly.
                    entry.avg_latency_ms = latency_ms
                else:
                    # Exponential moving average update.
                    entry.avg_latency_ms = (
                        alpha * latency_ms + (1 - alpha) * entry.avg_latency_ms
                    )
                break
