"""
gateway/observability package

Observability infrastructure for the AI Gateway.

Two complementary systems:
  langfuse_client.py — Structured distributed tracing via Langfuse.  Every
                       gateway request produces a Langfuse Trace with Spans
                       for each pipeline stage (rate-limit check, security
                       scan, cache lookup, provider call, contract evaluation).
                       This gives developers a full, searchable audit trail of
                       every LLM interaction.

  metrics.py         — Lightweight in-memory metrics aggregation.  Counters
                       and latency histograms are accumulated in-process and
                       exposed via the GET /metrics endpoint.  Designed for
                       quick dashboards without requiring an external metrics
                       backend.
"""
