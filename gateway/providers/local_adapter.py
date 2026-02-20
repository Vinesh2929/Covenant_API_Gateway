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

import json
import time
import uuid
from typing import Optional, AsyncIterator

from gateway.config import ProviderSettings
from gateway.providers.base import (
    BaseProviderAdapter,
    ProviderError,
    ProviderUnavailableError,
    TokenUsage,
)

# Ollama parameters that go inside an "options" sub-dict instead of the
# top-level request body.
_OLLAMA_OPTIONS_FIELDS = frozenset({
    "temperature", "top_p", "top_k", "num_predict", "num_ctx",
    "repeat_penalty", "seed", "tfs_z", "typical_p",
})

# Ollama done_reason → OpenAI finish_reason mapping.
_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
}


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
        url = f"{self._base_url}/api/chat"
        provider_body = self._translate_request(request)
        headers = self._build_headers()

        client = await self._get_client()

        try:
            # Use httpx streaming to consume the NDJSON response.
            # stream=True in the Ollama body means it sends one JSON object
            # per line as it generates tokens.
            async with client.stream(
                "POST",
                url,
                json=provider_body,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    body_bytes = await response.aread()
                    try:
                        error_body = json.loads(body_bytes)
                        error_msg = error_body.get("error", str(body_bytes))
                    except Exception:
                        error_msg = body_bytes.decode(errors="replace")

                    if response.status_code >= 500:
                        raise ProviderUnavailableError(
                            self.provider_name, response.status_code, str(error_msg)
                        )
                    raise ProviderError(
                        self.provider_name, response.status_code, str(error_msg)
                    )

                # Aggregate all streamed chunks into one final dict.
                aggregated = await self._aggregate_stream(response.aiter_lines())

        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError(
                self.provider_name, 503, f"Ollama connection error: {exc}"
            ) from exc

        return self._translate_response(aggregated)

    async def list_models(self) -> list[dict]:
        """
        Query Ollama for locally available models.

        Calls GET /api/tags which returns a list of pulled models with
        metadata (name, size, modified_at).  We reshape this into the
        OpenAI /v1/models response format for consistency.

        Returns:
            List of dicts in OpenAI /v1/models format.
        """
        client = await self._get_client()
        url = f"{self._base_url}/api/tags"

        try:
            response = await client.get(url, headers=self._build_headers())
        except Exception as exc:
            raise ProviderUnavailableError(
                self.provider_name, 503, f"Ollama not reachable: {exc}"
            ) from exc

        if response.status_code != 200:
            raise ProviderUnavailableError(
                self.provider_name, response.status_code, "Failed to list models"
            )

        data = response.json()
        models = data.get("models", [])

        # Transform Ollama model entries into OpenAI /v1/models format.
        return [
            {
                "id": model.get("name", ""),
                "object": "model",
                "owned_by": "local",
                "created": int(time.time()),
                "details": {
                    "size": model.get("size"),
                    "modified_at": model.get("modified_at"),
                },
            }
            for model in models
        ]

    def _build_headers(self) -> dict[str, str]:
        """
        Build minimal HTTP headers for the Ollama request.

        Ollama does not require authentication by default.  If an api_key was
        configured (e.g. for an Nginx-protected Ollama), include a Bearer token.

        Returns:
            Dict with Content-Type (and optional Authorization).
        """
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _translate_request(self, normalised: dict) -> dict:
        """
        Convert the normalised OpenAI-schema request to Ollama /api/chat format.

        Key differences from OpenAI:
          - model and messages stay as-is.
          - temperature/top_p/etc. move into an "options" sub-dict.
          - stream=True so we receive NDJSON (we aggregate it before returning).
          - max_tokens maps to options.num_predict (Ollama's token limit param).

        Args:
            normalised: Normalised chat completion request dict.

        Returns:
            Ollama /api/chat request body dict.
        """
        # Build the "options" block from any hyper-parameters in the request.
        options: dict = {}
        for field in _OLLAMA_OPTIONS_FIELDS:
            if field in normalised:
                options[field] = normalised[field]

        # max_tokens in OpenAI → num_predict in Ollama options.
        if "max_tokens" in normalised and "num_predict" not in options:
            options["num_predict"] = normalised["max_tokens"]

        body: dict = {
            "model": normalised["model"],
            "messages": normalised.get("messages", []),
            "stream": True,   # always stream; we aggregate in _aggregate_stream()
        }

        if options:
            body["options"] = options

        return body

    def _translate_response(self, aggregated: dict) -> dict:
        """
        Convert an aggregated Ollama response to the normalised OpenAI schema.

        The aggregated dict (built by _aggregate_stream) has:
          - "message": {"role": "assistant", "content": "<full text>"}
          - "prompt_eval_count": int  (prompt tokens)
          - "eval_count": int         (completion tokens)
          - "done_reason": str        (e.g. "stop")

        Args:
            aggregated: The assembled response dict from _aggregate_stream().

        Returns:
            Normalised OpenAI-schema response dict.
        """
        message = aggregated.get("message", {})
        prompt_tokens = aggregated.get("prompt_eval_count", 0)
        completion_tokens = aggregated.get("eval_count", 0)

        return {
            "id": f"local-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": aggregated.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content", ""),
                    },
                    "finish_reason": self._map_finish_reason(
                        aggregated.get("done_reason")
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def _aggregate_stream(self, line_iterator: AsyncIterator[str]) -> dict:
        """
        Read a streaming Ollama NDJSON response and aggregate it into a single
        response dict.

        Ollama sends one JSON object per line:
          - Non-final lines: {"model": "...", "message": {"role": "assistant",
            "content": "<token>"}, "done": false}
          - Final line:      {"model": "...", "message": {...}, "done": true,
            "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 50, ...}

        We accumulate the content fragments and keep the metadata from the
        final (done=true) line.

        Args:
            line_iterator: Async iterator of NDJSON lines from httpx.

        Returns:
            Aggregated dict with the full message content and usage metadata.
        """
        content_parts: list[str] = []
        final_chunk: dict = {}

        async for line in line_iterator:
            line = line.strip()
            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Each chunk has a message.content fragment to concatenate.
            msg = chunk.get("message", {})
            fragment = msg.get("content", "")
            if fragment:
                content_parts.append(fragment)

            # The final chunk (done=True) contains usage stats and stop reason.
            if chunk.get("done", False):
                final_chunk = chunk

        # Assemble the aggregated result with the full concatenated content.
        result = dict(final_chunk)
        result["message"] = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        return result

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
        return _FINISH_REASON_MAP.get(done_reason or "", "stop")
