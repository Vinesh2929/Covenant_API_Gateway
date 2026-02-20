"""
gateway/providers/anthropic_adapter.py

Adapter for the Anthropic Messages API (Claude models).

Responsibilities:
  - Translate from the gateway's canonical OpenAI-schema request format to
    Anthropic's Messages API format, which differs in several important ways:
      * The "system" prompt is a top-level field, not a message with role="system".
      * The model ID format is different (e.g. "claude-3-5-sonnet-20241022").
      * Token limits use "max_tokens" but the field is required, not optional.
      * The response shape uses "content" as a list of blocks, not a single string.
      * Stop reasons use different labels ("end_turn" vs "stop").
  - Translate Anthropic responses back to the normalised OpenAI format.
  - Set the required x-api-key and anthropic-version headers.
  - Map Anthropic error codes to the gateway's ProviderError hierarchy.
  - Handle Anthropic-specific features: prompt caching headers, beta features.

Key classes / functions:
  - AnthropicAdapter       — concrete adapter for Anthropic /v1/messages
    - __init__(settings)   — extract api_key and base_url from ProviderSettings
    - complete(request)    — async: translate → POST → translate response
    - _build_headers()     — x-api-key + anthropic-version headers
    - _translate_request() — OpenAI schema → Anthropic Messages API schema
    - _translate_response()— Anthropic response → normalised OpenAI schema
    - _extract_system()    — pull system message from messages list
    - _map_stop_reason()   — "end_turn" / "max_tokens" → "stop" / "length"
"""

from __future__ import annotations

from typing import Optional

from gateway.config import ProviderSettings
from gateway.providers.base import (
    BaseProviderAdapter,
    ProviderError,
    RateLimitError,
    ProviderUnavailableError,
)

# Anthropic API version pinned here — bump when adopting new API features.
ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicAdapter(BaseProviderAdapter):
    """
    Adapter for the Anthropic Claude Messages API.

    Handles the schema translation between the gateway's OpenAI-compatible
    canonical format and Anthropic's distinct Messages API format.

    Key translation differences:
      - System messages are extracted from the messages list and placed in a
        top-level "system" field.
      - The "max_tokens" field is required by Anthropic (defaults to 1024 if
        not specified in the incoming request).
      - Response content is returned as a list of typed blocks; we normalise
        this to a single string in the "choices[0].message.content" field.
      - Token usage field names match but stop reasons differ.
    """

    provider_name = "anthropic"

    # Anthropic API requires this header on every request.
    _API_VERSION = ANTHROPIC_API_VERSION

    def __init__(self, settings: ProviderSettings) -> None:
        """
        Extract Anthropic credentials from ProviderSettings.

        Args:
            settings: ProviderSettings with anthropic_api_key and
                      anthropic_base_url.
        """
        super().__init__(
            api_key=settings.anthropic_api_key or "",
            base_url=settings.anthropic_base_url,
            timeout=settings.request_timeout_seconds,
        )

    async def complete(self, request: dict) -> dict:
        """
        Forward a normalised chat completion request to the Anthropic API.

        Steps:
          1. Extract the system message via _extract_system().
          2. Translate the normalised request to Anthropic Messages format.
          3. POST to {base_url}/v1/messages with retry logic.
          4. Translate the Anthropic response back to the normalised format.

        Args:
            request: Normalised chat completion request dict (OpenAI schema).

        Returns:
            Normalised response dict (OpenAI schema).
        """
        # TODO: implement
        ...

    def _build_headers(self) -> dict[str, str]:
        """
        Build Anthropic-required authentication and version headers.

        Returns:
            Dict containing x-api-key, anthropic-version, and Content-Type.
        """
        # TODO: implement
        ...

    def _translate_request(self, normalised: dict) -> dict:
        """
        Convert the normalised OpenAI-schema request to Anthropic Messages format.

        Transformation rules:
          - Move system message (role="system") to top-level "system" field.
          - Keep all other messages as the "messages" list.
          - Ensure "max_tokens" is present (default 1024 if missing).
          - Map "stop" → "stop_sequences" if present.
          - Remove OpenAI-only fields: "n", "logprobs", "top_logprobs".

        Args:
            normalised: OpenAI-schema request dict.

        Returns:
            Anthropic Messages API request body dict.
        """
        # TODO: implement
        ...

    def _translate_response(self, provider_response: dict) -> dict:
        """
        Convert an Anthropic Messages response to the normalised OpenAI format.

        Transformation rules:
          - Concatenate all text blocks in "content" into a single string.
          - Wrap in choices[0].message.content structure.
          - Map "stop_reason" using _map_stop_reason().
          - Rename "input_tokens" → "prompt_tokens", "output_tokens" → "completion_tokens".

        Args:
            provider_response: Raw Anthropic API response dict.

        Returns:
            Normalised OpenAI-schema response dict.
        """
        # TODO: implement
        ...

    def _extract_system(self, messages: list[dict]) -> tuple[Optional[str], list[dict]]:
        """
        Separate the system message from the messages list.

        Args:
            messages: Full list of message dicts from the normalised request.

        Returns:
            A 2-tuple: (system_text_or_None, remaining_messages_without_system).
        """
        # TODO: implement — filter messages where role == "system"
        ...

    def _map_stop_reason(self, anthropic_stop_reason: Optional[str]) -> str:
        """
        Map an Anthropic stop reason to the OpenAI-compatible finish reason.

        Anthropic → OpenAI mapping:
          "end_turn"   → "stop"
          "max_tokens" → "length"
          "stop_sequence" → "stop"
          None         → "stop"

        Args:
            anthropic_stop_reason: The stop_reason string from the Anthropic response.

        Returns:
            An OpenAI-compatible finish_reason string.
        """
        # TODO: implement
        ...
