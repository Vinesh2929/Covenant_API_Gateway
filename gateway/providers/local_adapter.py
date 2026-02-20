"""
gateway/providers/local_adapter.py

Adapter for locally-hosted models served by Ollama.

Responsibilities:
  - Connect to an Ollama server (running as a Docker service or on the host).
  - Translate from the gateway's canonical OpenAI-schema request to the Ollama
    /api/chat endpoint format (which is close to OpenAI but has some differences).
  - Support the Ollama-specific "options" parameter block for model hyper-
    parameters (num_ctx, num_predict, temperature, top_p, etc.).
  - Handle the Ollama streaming JSON-lines response format, aggregating it
    into a single normalised response dict.
  - Check Ollama model availability at startup and surface clear errors when
    a requested model has not been pulled locally.
  - No API key required (local endpoint); add a configurable bearer token
    for deployments that put Nginx auth in front of Ollama.

Key classes / functions:
  - LocalAdapter           — concrete adapter for Ollama /api/chat
    - __init__(settings)   — set base_url from ProviderSettings.ollama_base_url
    - complete(request)    — async: translate → POST → aggregate → translate
    - list_models()        — async: GET /api/tags → list of available local models
    - _build_headers()     — minimal headers (no auth by default)
    - _translate_request() — OpenAI schema → Ollama /api/chat body
    - _translate_response()— Ollama response → normalised OpenAI schema
    - _aggregate_stream()  — async: read NDJSON stream, assemble full response
    - _map_finish_reason() — Ollama done_reason → OpenAI finish_reason
"""

from __future__ import annotations

from typing import Optional, AsyncIterator

from gateway.config import ProviderSettings
from gateway.providers.base import (
    BaseProviderAdapter,
    ProviderError,
    ProviderUnavailableError,
    TokenUsage,
)


class LocalAdapter(BaseProviderAdapter):
    """
    Adapter for locally-hosted models via the Ollama REST API.

    Ollama exposes a /api/chat endpoint that closely mirrors OpenAI's format
    but with key differences:
      - Responses can be streamed as newline-delimited JSON (NDJSON).
      - Model names are simple strings like "llama3", "mistral", "codellama".
      - Token counts come from an "eval_count" / "prompt_eval_count" field
        rather than a "usage" object.
      - There is no "finish_reason" field; "done" and "done_reason" are used.
      - Temperature and other params go inside an "options" sub-object.

    This adapter always uses stream=True and aggregates the NDJSON response
    before returning, giving callers a uniform non-streaming interface.
    """

    provider_name = "local"

    def __init__(self, settings: ProviderSettings) -> None:
        """
        Configure the adapter to connect to the Ollama server.

        Args:
            settings: ProviderSettings with ollama_base_url and
                      request_timeout_seconds.
        """
        super().__init__(
            api_key="",                           # Ollama has no API key by default
            base_url=settings.ollama_base_url,
            timeout=settings.request_timeout_seconds,
        )

    async def complete(self, request: dict) -> dict:
        """
        Forward a normalised chat request to the Ollama /api/chat endpoint.

        Steps:
          1. Translate normalised request to Ollama format.
          2. POST to {ollama_base_url}/api/chat with stream=True.
          3. Aggregate the NDJSON stream via _aggregate_stream().
          4. Translate the aggregated response to normalised format.

        Args:
            request: Normalised OpenAI-schema chat completion request.

        Returns:
            Normalised OpenAI-schema response dict.
        """
        # TODO: implement
        ...

    async def list_models(self) -> list[dict]:
        """
        Query Ollama for locally available models.

        Returns:
            List of dicts in OpenAI /v1/models format, derived from the
            Ollama GET /api/tags response.
        """
        # TODO: implement — GET {base_url}/api/tags, transform response
        ...

    def _build_headers(self) -> dict[str, str]:
        """
        Build minimal HTTP headers for the Ollama request.

        Returns:
            Dict with Content-Type only (no auth unless api_key is set).
        """
        # TODO: implement
        ...

    def _translate_request(self, normalised: dict) -> dict:
        """
        Convert the normalised OpenAI-schema request to Ollama /api/chat format.

        Transformation rules:
          - Keep "model" and "messages" as-is.
          - Move "temperature", "top_p", "num_predict" into an "options" sub-dict.
          - Set stream=True so we receive an NDJSON response.
          - Remove OpenAI-only fields Ollama does not accept.

        Args:
            normalised: Normalised chat completion request dict.

        Returns:
            Ollama /api/chat request body dict.
        """
        # TODO: implement
        ...

    def _translate_response(self, aggregated: dict) -> dict:
        """
        Convert an aggregated Ollama response to the normalised OpenAI schema.

        Transformation rules:
          - The "message" field maps to choices[0].message.
          - "prompt_eval_count" → usage.prompt_tokens.
          - "eval_count"        → usage.completion_tokens.
          - _map_finish_reason("done_reason") → choices[0].finish_reason.

        Args:
            aggregated: The assembled response dict from _aggregate_stream().

        Returns:
            Normalised OpenAI-schema response dict.
        """
        # TODO: implement
        ...

    async def _aggregate_stream(self, response_stream) -> dict:
        """
        Read a streaming Ollama NDJSON response and aggregate it into a single
        response dict.

        Ollama sends one JSON object per line.  Each line has a "message.content"
        fragment and the final line has "done": true with the full usage counts.

        Args:
            response_stream: An async iterable of bytes/lines from httpx.

        Returns:
            Aggregated dict with the full message content and usage metadata.
        """
        # TODO: implement — async for line in response_stream: json.loads(line), accumulate
        ...

    def _map_finish_reason(self, done_reason: Optional[str]) -> str:
        """
        Map an Ollama done_reason to an OpenAI-compatible finish_reason.

        Ollama → OpenAI mapping:
          "stop"        → "stop"
          "length"      → "length"
          None / other  → "stop"

        Args:
            done_reason: The done_reason string from the final Ollama stream chunk.

        Returns:
            An OpenAI-compatible finish_reason string.
        """
        # TODO: implement
        ...
