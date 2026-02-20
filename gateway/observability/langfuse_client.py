"""
gateway/observability/langfuse_client.py

Thin wrapper around the official Langfuse Python SDK.

Responsibilities:
  - Initialise a Langfuse client from settings and expose a clean interface
    that hides Langfuse-specific types from the rest of the codebase.
  - Create a top-level Trace for every proxied request, tagged with app_id,
    provider, model, and request metadata.
  - Attach child Spans for each pipeline stage so engineers can see exactly
    how much time was spent on: rate limiting, security scanning, cache lookup,
    provider inference, and contract evaluation.
  - Record the final token usage (prompt_tokens, completion_tokens, total_tokens)
    and cost estimate as Langfuse generation metadata.
  - Support disabling tracing entirely (enabled=False in settings) so the
    gateway can run without Langfuse credentials in local dev mode.
  - Provide a flush() method for clean shutdown so no traces are dropped.

Key classes / functions:
  - GatewayTrace             — context manager that wraps a single proxied request
  - LangfuseClient           — initialises the SDK and creates GatewayTrace objects
    - __init__(settings)     — configure client; no-op if enabled=False
    - create_trace(...)      — factory: returns a GatewayTrace context manager
    - flush()                — flush pending events before shutdown
  - GatewayTrace (context manager):
    - __enter__ / __aenter__ — start the Langfuse trace
    - span(name, ...)        — start a child span (returns a context manager)
    - record_generation(...) — record provider call metadata (tokens, model, cost)
    - set_output(...)        — attach the final response to the trace
    - __exit__ / __aexit__   — end the trace, mark success or error
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from gateway.config import LangfuseSettings


# ---------------------------------------------------------------------------
# Trace context manager
# ---------------------------------------------------------------------------

class GatewayTrace:
    """
    Async context manager representing a single end-to-end gateway request
    trace in Langfuse.

    Obtain an instance from LangfuseClient.create_trace() — do not instantiate
    directly.

    Usage::

        async with client.create_trace(request_id="abc", app_id="my-app") as trace:
            async with trace.span("security_scan"):
                result = await security_guard.scan(prompt)
            async with trace.span("provider_call", model="gpt-4o"):
                response = await adapter.complete(request)
            trace.record_generation(
                model="gpt-4o",
                prompt_tokens=150,
                completion_tokens=300,
            )
    """

    def __init__(
        self,
        client: "LangfuseClient",
        request_id: str,
        app_id: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Store trace parameters.  The Langfuse trace object is created in
        __aenter__.

        Args:
            client:     The parent LangfuseClient instance.
            request_id: Unique ID for this gateway request (added as X-Request-ID).
            app_id:     Application identifier for grouping in the Langfuse UI.
            metadata:   Arbitrary key/value pairs added as trace-level metadata.
        """
        self._client = client
        self._request_id = request_id
        self._app_id = app_id
        self._metadata = metadata or {}
        self._trace = None  # langfuse.Trace, set in __aenter__

    async def __aenter__(self) -> "GatewayTrace":
        """Start the Langfuse trace."""
        # TODO: implement — self._trace = self._client._langfuse.trace(...)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """End the trace, marking it as errored if an exception occurred."""
        # TODO: implement — set trace status, call self._trace.update(...)
        ...

    @asynccontextmanager
    async def span(self, name: str, **metadata: Any) -> AsyncIterator[None]:
        """
        Create a child span for a named pipeline stage.

        Args:
            name:     Human-readable span name (e.g. "rate_limit_check").
            **metadata: Additional key/value pairs attached to the span.

        Yields:
            None (the span is ended automatically on exit).
        """
        # TODO: implement — create span, yield, end span in finally block
        ...

    def record_generation(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: Optional[float] = None,
    ) -> None:
        """
        Attach LLM generation metadata to the trace.

        Called after the provider adapter returns a response so that Langfuse
        can calculate per-trace cost and token usage.

        Args:
            model:             The exact model ID used by the provider.
            prompt_tokens:     Tokens in the input prompt.
            completion_tokens: Tokens in the generated response.
            cost_usd:          Optional estimated cost in USD.
        """
        # TODO: implement — self._trace.generation(...)
        ...

    def set_output(self, response: dict) -> None:
        """
        Attach the final normalised response as the trace output.

        Args:
            response: The normalised response dict returned to the client.
        """
        # TODO: implement — self._trace.update(output=response)
        ...

    def set_error(self, error: Exception) -> None:
        """
        Mark the trace as failed and attach error details.

        Args:
            error: The exception that caused the request to fail.
        """
        # TODO: implement
        ...


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LangfuseClient:
    """
    Initialises the Langfuse SDK and creates GatewayTrace instances.

    When tracing is disabled (settings.langfuse.enabled == False), all methods
    return no-op stubs so the rest of the codebase can call them unconditionally
    without conditional logic.
    """

    def __init__(self, settings: LangfuseSettings) -> None:
        """
        Initialise the Langfuse client from settings.

        If enabled=False or the keys are missing, self._langfuse is set to None
        and all subsequent calls are no-ops.

        Args:
            settings: LangfuseSettings with public_key, secret_key, host.
        """
        self._settings = settings
        self._langfuse = None  # langfuse.Langfuse instance, or None if disabled
        # TODO: implement — if settings.enabled: import langfuse; self._langfuse = Langfuse(...)

    def create_trace(
        self,
        request_id: str,
        app_id: str,
        metadata: Optional[dict] = None,
    ) -> GatewayTrace:
        """
        Factory method that returns a GatewayTrace context manager.

        Args:
            request_id: Unique request identifier (UUID or custom).
            app_id:     Application identifier for the Langfuse project.
            metadata:   Trace-level metadata (model, provider, IP, etc.).

        Returns:
            A GatewayTrace instance ready to be used as an async context manager.
        """
        # TODO: implement
        ...

    def flush(self) -> None:
        """
        Block until all pending Langfuse events have been sent.

        Called during FastAPI shutdown to prevent trace loss.
        """
        # TODO: implement — self._langfuse.flush() if self._langfuse
        ...
