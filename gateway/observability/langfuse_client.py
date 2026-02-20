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

import time
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
        """
        Start the Langfuse trace.

        When Langfuse is enabled, creates a trace tagged with the request ID
        and app ID so all child spans are grouped in the Langfuse UI.

        When disabled (self._client._langfuse is None), this is a no-op — the
        rest of the code can still call span() and record_generation() safely.
        """
        lf = self._client._langfuse
        if lf is not None:
            # lf.trace() registers the trace with Langfuse.
            # - name: displayed in the Langfuse UI as the trace label
            # - user_id: the app_id helps group traces by application
            # - id: the request_id makes traces searchable by request
            # - metadata: any extra context (model, provider, etc.)
            self._trace = lf.trace(
                name="gateway-request",
                user_id=self._app_id,
                id=self._request_id,
                metadata=self._metadata,
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        End the trace, marking it as errored if an exception occurred.

        Langfuse traces don't have an explicit "end" call — they're
        finalised when all spans complete.  We use update() to attach
        error context when the request fails.
        """
        if self._trace is not None and exc_type is not None:
            # An unhandled exception escaped the `async with` block.
            # Attach error metadata so the trace is marked failed in Langfuse.
            self._trace.update(
                status_message=f"{exc_type.__name__}: {exc_val}",
                level="ERROR",
            )
        # Returning None (falsy) does NOT suppress the exception.

    @asynccontextmanager
    async def span(self, name: str, **metadata: Any) -> AsyncIterator[None]:
        """
        Create a child span for a named pipeline stage.

        The span automatically records its start and end time, so latency
        for each pipeline stage appears in the Langfuse trace waterfall view.

        Args:
            name:     Human-readable span name (e.g. "rate_limit_check").
            **metadata: Additional key/value pairs attached to the span.

        Yields:
            None (the span is ended automatically on exit).

        Usage::

            async with trace.span("security_scan", model="gpt-4o"):
                result = await guard.scan(prompt)
        """
        if self._trace is None:
            # Tracing disabled — yield immediately without creating a span.
            yield
            return

        # Create child span on the trace.  Langfuse records the start time
        # automatically when span() is called.
        span = self._trace.span(
            name=name,
            metadata=metadata if metadata else None,
        )
        start = time.perf_counter()
        try:
            yield
        finally:
            # Always end the span, even if the body raised an exception.
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            span.end(metadata={"latency_ms": elapsed_ms})

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
        if self._trace is None:
            return

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        # generation() creates a Langfuse Generation object — a specialised
        # span for LLM calls that also tracks token counts and estimated cost.
        kwargs: dict[str, Any] = {
            "name": "llm-call",
            "model": model,
            "usage": usage,
        }
        if cost_usd is not None:
            kwargs["metadata"] = {"cost_usd": cost_usd}

        self._trace.generation(**kwargs)

    def set_output(self, response: dict) -> None:
        """
        Attach the final normalised response as the trace output.

        The output appears in the Langfuse UI alongside the trace, making it
        easy to inspect what the model returned for a given request.

        Args:
            response: The normalised response dict returned to the client.
        """
        if self._trace is None:
            return

        # Extract the assistant message content from the normalised response.
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = str(response)

        self._trace.update(output=content)

    def set_error(self, error: Exception) -> None:
        """
        Mark the trace as failed and attach error details.

        Called explicitly when we catch a known error (e.g. ProviderUnavailableError)
        and want to record it without letting the exception propagate to __aexit__.

        Args:
            error: The exception that caused the request to fail.
        """
        if self._trace is None:
            return

        self._trace.update(
            status_message=f"{type(error).__name__}: {error}",
            level="ERROR",
        )


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

        The langfuse import is deferred here so the gateway can start without
        the langfuse package installed if tracing is disabled.

        Args:
            settings: LangfuseSettings with public_key, secret_key, host.
        """
        self._settings = settings
        self._langfuse = None  # langfuse.Langfuse instance, or None if disabled

        if settings.enabled and settings.public_key and settings.secret_key:
            # Deferred import — only load the Langfuse SDK when actually needed.
            # This prevents an ImportError from crashing the gateway when
            # langfuse is not installed in development.
            try:
                from langfuse import Langfuse  # type: ignore[import]

                self._langfuse = Langfuse(
                    public_key=settings.public_key,
                    secret_key=settings.secret_key,
                    host=settings.host,
                    # flush_interval: how often background thread sends events.
                    # Lower = more timely traces; higher = fewer API calls.
                    flush_interval=getattr(settings, "flush_interval", 0.5),
                )
            except ImportError:
                # Graceful degradation: if langfuse is not installed, tracing
                # simply does nothing.  The gateway continues to serve requests.
                pass

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
        # Always return a GatewayTrace — when self._langfuse is None, all
        # GatewayTrace methods are no-ops, so callers never need to check.
        return GatewayTrace(
            client=self,
            request_id=request_id,
            app_id=app_id,
            metadata=metadata,
        )

    def flush(self) -> None:
        """
        Block until all pending Langfuse events have been sent.

        Called during FastAPI shutdown to prevent trace loss.  Without this,
        the Langfuse background thread may not have flushed its queue before
        the process exits.
        """
        if self._langfuse is not None:
            self._langfuse.flush()
