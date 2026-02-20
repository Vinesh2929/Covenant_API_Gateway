"""
gateway/providers/base.py

Abstract base class (and Protocol) for all provider adapters.

Responsibilities:
  - Define the contract that every provider adapter must fulfil so that the
    router and main request pipeline can treat all providers uniformly.
  - Declare the normalised request and response formats used internally by the
    gateway (OpenAI chat completions schema as the canonical format).
  - Provide shared utility methods for HTTP retries and error classification
    that all concrete adapters inherit.

The normalised request format is the standard OpenAI /v1/chat/completions JSON:
    {
      "model": "...",
      "messages": [{"role": "user", "content": "..."}],
      "temperature": 0.7,
      "max_tokens": 1024,
      ...
    }

The normalised response format is also OpenAI-compatible:
    {
      "id": "...",
      "object": "chat.completion",
      "created": 1700000000,
      "model": "...",
      "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 150, "completion_tokens": 300, "total_tokens": 450}
    }

Key classes / functions:
  - ProviderError            — base exception for provider-side errors
  - RateLimitError           — 429 from upstream provider
  - ProviderUnavailableError — 5xx from upstream provider (retryable)
  - TokenUsage               — dataclass: prompt_tokens, completion_tokens, total
  - BaseProviderAdapter      — abstract base class all adapters must subclass
    - complete(request)      — async abstract: send request, return normalised response
    - _build_headers()       — abstract: construct provider-specific auth headers
    - _translate_request()   — abstract: normalised → provider-native format
    - _translate_response()  — abstract: provider-native → normalised format
    - _request_with_retry()  — concrete: HTTP POST with exponential backoff
    - extract_usage()        — concrete: pull token counts from normalised response
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import httpx


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """
    Base exception for all errors originating from an upstream LLM provider.

    Attributes:
        provider:    Name of the provider that raised the error.
        status_code: HTTP status code from the provider response.
        message:     Error message from the provider API.
    """
    def __init__(self, provider: str, status_code: int, message: str) -> None:
        self.provider = provider
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{provider}] HTTP {status_code}: {message}")


class RateLimitError(ProviderError):
    """
    Raised when the upstream provider returns HTTP 429 (too many requests).

    Includes the Retry-After value from the provider's response headers when
    available.

    Attributes:
        retry_after: Seconds to wait before retrying, or None if not specified.
    """
    def __init__(self, provider: str, retry_after: Optional[float] = None) -> None:
        self.retry_after = retry_after
        super().__init__(provider, 429, "Provider rate limit exceeded")


class ProviderUnavailableError(ProviderError):
    """
    Raised when the upstream provider returns a 5xx response.

    These errors are considered transient and the gateway will retry with
    exponential backoff up to the configured maximum.
    """
    pass


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    """
    Token consumption data extracted from a provider response.

    Attributes:
        prompt_tokens:      Tokens used in the input messages.
        completion_tokens:  Tokens in the generated response.
        total_tokens:       Sum of prompt + completion tokens.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# ---------------------------------------------------------------------------
# Abstract base adapter
# ---------------------------------------------------------------------------

class BaseProviderAdapter(ABC):
    """
    Abstract base class for all LLM provider adapters.

    Concrete adapters (OpenAI, Anthropic, Local) inherit from this class and
    implement the three abstract methods: complete(), _build_headers(), and
    _translate_request() / _translate_response().

    Shared infrastructure (HTTP retries, usage extraction) lives here and is
    available to all adapters without duplication.
    """

    #: Provider name used in logging and metrics (override in subclasses).
    provider_name: str = "base"

    #: Maximum number of retry attempts for transient (5xx) errors.
    max_retries: int = 3

    #: Base delay in seconds for the first retry (doubles each attempt).
    retry_base_delay: float = 1.0

    def __init__(self, api_key: str, base_url: str, timeout: int = 60) -> None:
        """
        Initialise the adapter with credentials and a shared HTTP client.

        Args:
            api_key:  The provider API key (may be empty for local adapters).
            base_url: The base URL for the provider's API.
            timeout:  Per-request timeout in seconds.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Return a shared async HTTP client, creating it on first call.

        We share one client across all requests to benefit from connection
        pooling — httpx keeps keep-alive connections open so subsequent
        requests to the same host avoid TCP handshake overhead.

        Returns:
            An httpx.AsyncClient configured with the adapter's timeout.
        """
        if self._client is None:
            # httpx.Timeout splits the timeout into connect, read, write, pool.
            # Passing a single float sets all four to the same value.
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                # http2=True speeds up multiplexed requests to providers that
                # support HTTP/2 (Anthropic, OpenAI both do).
                http2=True,
            )
        return self._client

    @abstractmethod
    async def complete(self, request: dict) -> dict:
        """
        Send a normalised chat completion request to the provider and return
        a normalised response.

        This is the primary method called by the gateway's request pipeline.

        Args:
            request: Normalised chat completion request dict (OpenAI schema).

        Returns:
            Normalised response dict (OpenAI schema).

        Raises:
            RateLimitError:          If the provider returns HTTP 429.
            ProviderUnavailableError: If the provider returns a 5xx response.
            ProviderError:            For any other provider-side error.
        """
        ...

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        """
        Construct the HTTP request headers for this provider.

        Returns:
            A dict of header name → value pairs including authentication.
        """
        ...

    @abstractmethod
    def _translate_request(self, normalised: dict) -> dict:
        """
        Convert a normalised (OpenAI-schema) request dict into the provider's
        native request format.

        Args:
            normalised: Normalised chat completion request dict.

        Returns:
            Provider-native request body dict ready to JSON-serialise.
        """
        ...

    @abstractmethod
    def _translate_response(self, provider_response: dict) -> dict:
        """
        Convert a provider-native response dict into the normalised
        (OpenAI-schema) format.

        Args:
            provider_response: Raw JSON response body from the provider API.

        Returns:
            Normalised response dict.
        """
        ...

    async def _request_with_retry(
        self,
        url: str,
        body: dict,
        headers: dict,
    ) -> dict:
        """
        Execute an HTTP POST to `url` with exponential backoff on transient errors.

        Retries up to self.max_retries times on 5xx responses.  Does NOT retry
        on 4xx errors (including 429 — let the caller decide on retry logic).

        Exponential backoff formula:
            delay = retry_base_delay * (2 ** attempt)
            attempt 0 → 1s, attempt 1 → 2s, attempt 2 → 4s

        Args:
            url:     Full URL including path.
            body:    Request body dict (will be JSON-serialised).
            headers: HTTP headers dict.

        Returns:
            Parsed JSON response body dict.

        Raises:
            RateLimitError:           On HTTP 429.
            ProviderUnavailableError: On HTTP 5xx after max_retries exhausted.
            ProviderError:            On other HTTP error codes.
        """
        client = await self._get_client()
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = await client.post(url, json=body, headers=headers)
            except httpx.TimeoutException as exc:
                # Network-level timeout — treat like a 503, eligible for retry.
                last_error = ProviderUnavailableError(
                    self.provider_name, 503, f"Request timeout: {exc}"
                )
                if attempt < self.max_retries - 1:
                    delay = self.retry_base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                continue

            # --- Success path ---
            if response.status_code == 200:
                return response.json()

            # --- Error paths ---
            # Try to parse the error body; fall back to raw text.
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": {"message": response.text}}

            if response.status_code == 429:
                # Rate limited by the upstream provider.
                # Extract Retry-After if present.
                retry_after_header = response.headers.get("retry-after")
                retry_after = float(retry_after_header) if retry_after_header else None
                raise RateLimitError(self.provider_name, retry_after)

            if response.status_code >= 500:
                # Transient server error — eligible for retry with backoff.
                last_error = ProviderUnavailableError(
                    self.provider_name,
                    response.status_code,
                    str(error_body.get("error", {}).get("message", "Server error")),
                )
                if attempt < self.max_retries - 1:
                    delay = self.retry_base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                continue

            # 4xx (not 429) — client error, do not retry.
            msg = str(error_body.get("error", {}).get("message", "Unknown error"))
            raise ProviderError(self.provider_name, response.status_code, msg)

        # All retries exhausted for 5xx errors.
        raise last_error or ProviderUnavailableError(
            self.provider_name, 503, "All retry attempts exhausted"
        )

    def extract_usage(self, normalised_response: dict) -> TokenUsage:
        """
        Extract token usage from a normalised response dict.

        The normalised format always has a "usage" key (guaranteed by
        _translate_response() in each adapter).  This helper exists so
        main.py doesn't need to know the dict structure.

        Args:
            normalised_response: A response dict in OpenAI schema format.

        Returns:
            A TokenUsage dataclass with prompt, completion, and total token counts.
        """
        usage = normalised_response.get("usage", {})
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", prompt + completion)
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )

    async def close(self) -> None:
        """
        Close the underlying HTTP client and release connections.

        Should be called during the FastAPI shutdown lifespan hook.
        Without this, the event loop emits "Unclosed client session" warnings.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None
