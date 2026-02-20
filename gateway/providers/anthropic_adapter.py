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

import time
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

# Default max_tokens when the client request does not specify one.
# Anthropic requires max_tokens; OpenAI treats it as optional.
_DEFAULT_MAX_TOKENS = 1024

# OpenAI-only fields that have no equivalent in the Anthropic API.
_OPENAI_ONLY_FIELDS = frozenset({
    "n", "logprobs", "top_logprobs", "presence_penalty",
    "frequency_penalty", "best_of", "logit_bias",
    "app_id", "x_provider", "x_request_id",
})

# Mapping from Anthropic stop reasons to OpenAI finish reasons.
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
}


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
          1. Translate the normalised request to Anthropic Messages format.
          2. POST to {base_url}/v1/messages with retry logic.
          3. Translate the Anthropic response back to the normalised format.

        Args:
            request: Normalised chat completion request dict (OpenAI schema).

        Returns:
            Normalised response dict (OpenAI schema).
        """
        # The Anthropic base_url is typically "https://api.anthropic.com".
        # The endpoint is /v1/messages (not /v1/chat/completions).
        url = f"{self._base_url}/v1/messages"
        provider_body = self._translate_request(request)
        headers = self._build_headers()

        raw_response = await self._request_with_retry(url, provider_body, headers)

        return self._translate_response(raw_response)

    def _build_headers(self) -> dict[str, str]:
        """
        Build Anthropic-required authentication and version headers.

        Anthropic uses x-api-key instead of the Authorization: Bearer pattern.
        The anthropic-version header pins the API behaviour — omitting it or
        sending an old version may cause unexpected response formats.

        Returns:
            Dict containing x-api-key, anthropic-version, and Content-Type.
        """
        return {
            "x-api-key": self._api_key,
            "anthropic-version": self._API_VERSION,
            "Content-Type": "application/json",
        }

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
        messages = normalised.get("messages", [])

        # Extract system message and remaining conversation messages.
        system_text, remaining_messages = self._extract_system(messages)

        body: dict = {
            "model": normalised["model"],
            "messages": remaining_messages,
            # max_tokens is required by Anthropic; default if omitted.
            "max_tokens": normalised.get("max_tokens", _DEFAULT_MAX_TOKENS),
        }

        # Only include optional fields if they were provided in the request.
        if system_text:
            body["system"] = system_text

        if "temperature" in normalised:
            body["temperature"] = normalised["temperature"]

        if "top_p" in normalised:
            body["top_p"] = normalised["top_p"]

        # OpenAI uses "stop" as a string or list; Anthropic uses "stop_sequences" (list).
        stop = normalised.get("stop")
        if stop:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        return body

    def _translate_response(self, provider_response: dict) -> dict:
        """
        Convert an Anthropic Messages response to the normalised OpenAI format.

        Anthropic response shape::

            {
              "id": "msg_...",
              "type": "message",
              "role": "assistant",
              "content": [{"type": "text", "text": "Hello!"}],
              "model": "claude-3-5-sonnet-20241022",
              "stop_reason": "end_turn",
              "usage": {"input_tokens": 10, "output_tokens": 20}
            }

        Args:
            provider_response: Raw Anthropic API response dict.

        Returns:
            Normalised OpenAI-schema response dict.
        """
        # Concatenate all text content blocks into a single string.
        # Anthropic returns content as a list of typed blocks; we flatten them.
        content_blocks = provider_response.get("content", [])
        content_text = "".join(
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        )

        usage_raw = provider_response.get("usage", {})
        # Anthropic uses input_tokens / output_tokens; we normalise to OpenAI names.
        prompt_tokens = usage_raw.get("input_tokens", 0)
        completion_tokens = usage_raw.get("output_tokens", 0)

        return {
            "id": provider_response.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": provider_response.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content_text,
                    },
                    "finish_reason": self._map_stop_reason(
                        provider_response.get("stop_reason")
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _extract_system(self, messages: list[dict]) -> tuple[Optional[str], list[dict]]:
        """
        Separate the system message from the messages list.

        Anthropic requires the system prompt as a top-level field, not as a
        message with role="system".  This method splits the messages list into
        the system text and the remaining conversation.

        Args:
            messages: Full list of message dicts from the normalised request.

        Returns:
            A 2-tuple: (system_text_or_None, remaining_messages_without_system).
        """
        system_parts: list[str] = []
        remaining: list[dict] = []

        for msg in messages:
            if msg.get("role") == "system":
                # Multiple system messages are concatenated with a newline.
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Multi-part content blocks — extract text parts.
                    text = " ".join(
                        part.get("text", "")
                        for part in content
                        if part.get("type") == "text"
                    )
                    system_parts.append(text)
                else:
                    system_parts.append(str(content))
            else:
                remaining.append(msg)

        system_text = "\n".join(system_parts) if system_parts else None
        return system_text, remaining

    def _map_stop_reason(self, anthropic_stop_reason: Optional[str]) -> str:
        """
        Map an Anthropic stop reason to the OpenAI-compatible finish reason.

        Anthropic → OpenAI mapping:
          "end_turn"      → "stop"
          "max_tokens"    → "length"
          "stop_sequence" → "stop"
          None            → "stop"

        Args:
            anthropic_stop_reason: The stop_reason string from the Anthropic response.

        Returns:
            An OpenAI-compatible finish_reason string.
        """
        return _STOP_REASON_MAP.get(anthropic_stop_reason or "", "stop")
