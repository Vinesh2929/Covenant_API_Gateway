"""
gateway/providers/openai_adapter.py

Adapter for the OpenAI Chat Completions API (and compatible endpoints such as
Azure OpenAI, Groq, Together AI, Fireworks AI, etc.).

Responsibilities:
  - Implement the three abstract methods of BaseProviderAdapter for the OpenAI
    wire format (which is already the gateway's canonical format, so translation
    is mostly a pass-through with a few field adjustments).
  - Add the correct Authorization header (Bearer token).
  - Handle OpenAI-specific error response shapes and map them to the gateway's
    ProviderError hierarchy.
  - Support optional organisation ID header (OpenAI-Organization).
  - Support Azure OpenAI endpoints, which use a different URL scheme and an
    api-version query parameter instead of a Bearer token.

Key classes / functions:
  - OpenAIAdapter          — concrete adapter for OpenAI /v1/chat/completions
    - __init__(settings)   — extract api_key, base_url from ProviderSettings
    - complete(request)    — async: call _request_with_retry, translate response
    - _build_headers()     — Bearer auth + optional Org header
    - _translate_request() — mostly pass-through; remove gateway-only fields
    - _translate_response()— identity (already OpenAI format); ensure all keys exist
    - _parse_error()       — map OpenAI error JSON to ProviderError subtype
"""

from __future__ import annotations

from gateway.config import ProviderSettings
from gateway.providers.base import (
    BaseProviderAdapter,
    ProviderError,
    RateLimitError,
    ProviderUnavailableError,
)


class OpenAIAdapter(BaseProviderAdapter):
    """
    Adapter for the OpenAI /v1/chat/completions endpoint.

    Because the gateway uses OpenAI's schema as its canonical internal format,
    translation is minimal — the main work here is authentication headers,
    error mapping, and stripping any gateway-internal fields before forwarding.

    Also works as a drop-in adapter for any OpenAI-compatible API (Azure,
    Groq, Together AI, etc.) by simply changing the base_url and api_key.

    Usage (via router, not directly)::

        adapter = OpenAIAdapter(settings.providers)
        response = await adapter.complete(normalised_request)
    """

    provider_name = "openai"

    def __init__(self, settings: ProviderSettings) -> None:
        """
        Extract OpenAI credentials from ProviderSettings.

        Args:
            settings: The ProviderSettings slice with openai_api_key and
                      openai_base_url.
        """
        super().__init__(
            api_key=settings.openai_api_key or "",
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout_seconds,
        )
        # TODO: store org_id if present in settings

    async def complete(self, request: dict) -> dict:
        """
        Forward a normalised chat completion request to the OpenAI API.

        Steps:
          1. Translate the normalised request to OpenAI format.
          2. POST to {base_url}/chat/completions with retry logic.
          3. Translate the response back to the normalised format.

        Args:
            request: Normalised chat completion request dict.

        Returns:
            Normalised response dict.
        """
        # TODO: implement
        ...

    def _build_headers(self) -> dict[str, str]:
        """
        Build OpenAI authentication headers.

        Returns:
            Dict with Authorization: Bearer <api_key> and Content-Type.
            Includes OpenAI-Organization if an org ID was configured.
        """
        # TODO: implement
        ...

    def _translate_request(self, normalised: dict) -> dict:
        """
        Prepare the request body for the OpenAI API.

        Since OpenAI format is the gateway's canonical format, this mostly
        strips gateway-internal metadata fields (e.g. "app_id", "x_provider")
        that should not be forwarded upstream.

        Args:
            normalised: Normalised request dict.

        Returns:
            Clean OpenAI-compatible request body.
        """
        # TODO: implement — copy dict, remove gateway-internal keys
        ...

    def _translate_response(self, provider_response: dict) -> dict:
        """
        Ensure the OpenAI response conforms to the canonical normalised format.

        Adds any missing fields with defaults so downstream consumers can
        rely on a consistent schema.

        Args:
            provider_response: Raw JSON from the OpenAI API.

        Returns:
            Normalised response dict with guaranteed key presence.
        """
        # TODO: implement — set defaults for missing optional fields
        ...

    def _parse_error(self, status_code: int, body: dict) -> ProviderError:
        """
        Map an OpenAI error response to the appropriate ProviderError subtype.

        OpenAI error body shape::

            {"error": {"message": "...", "type": "...", "code": "..."}}

        Args:
            status_code: HTTP status code from the OpenAI response.
            body:        Parsed JSON error body.

        Returns:
            An appropriate ProviderError subclass instance.
        """
        # TODO: implement — check status_code, check error.type for rate limits
        ...
