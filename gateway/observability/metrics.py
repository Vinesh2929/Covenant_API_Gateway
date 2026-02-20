"""
gateway/observability/metrics.py

In-memory metrics aggregation for the AI Gateway.

Responsibilities:
  - Maintain lock-protected counters and latency histograms that are updated
    on every request without touching an external service.
  - Compute latency percentiles (p50, p95, p99) from a bounded circular buffer
    of recent latency samples.
  - Expose a snapshot() method that serialises all metrics into a JSON-safe
    dict for the GET /metrics endpoint.
  - Support per-provider and per-model breakdowns so operators can see which
    upstream is contributing most to latency or errors.
  - Designed to be a singleton shared across the entire process; thread-safe
    via threading.Lock for counter updates.

Key classes / functions:
  - LatencyBuffer           — fixed-size circular buffer for p-tile calculations
  - ProviderStats           — dataclass: per-provider counters and latency buffer
  - MetricsCollector        — singleton class owning all counters
    - __init__()            — initialise all counters to zero
    - record_request(...)   — called after each proxied request
    - record_cache_hit()    — increment cache hit counter
    - record_cache_miss()   — increment cache miss counter
    - record_injection_blocked(tier) — increment security block counter
    - record_rate_limited() — increment 429 counter
    - record_provider_error(provider) — increment per-provider error counter
    - snapshot()            — return all metrics as a JSON-safe dict
    - reset()               — reset all counters (used in tests)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Latency buffer
# ---------------------------------------------------------------------------

class LatencyBuffer:
    """
    Thread-safe fixed-size circular buffer for latency samples.

    WHY A CIRCULAR BUFFER?
      We want to answer "what is the p95 latency over the last N requests?"
      A circular buffer (deque with maxlen) automatically discards the oldest
      sample when full, so we always have the N most recent values without
      ever growing unboundedly in memory.

    WHY NOT A HISTOGRAM?
      Histograms (like Prometheus uses) are more memory-efficient but require
      pre-defined bucket boundaries.  A sorted sample buffer lets us compute
      arbitrary percentiles exactly, which is useful during development when
      you don't know your latency distribution yet.

    Thread safety:
      We protect add() and percentile() with a threading.Lock because the
      FastAPI event loop runs in one thread but MetricsCollector.record_*
      methods could theoretically be called from worker threads too.

    Args:
        maxlen: Maximum latency samples to retain. Older ones are evicted
                automatically when the buffer is full. Default 1000.
    """

    def __init__(self, maxlen: int = 1000) -> None:
        # deque(maxlen=N) is Python's built-in circular buffer.
        # Appending to a full deque automatically removes the oldest element.
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, latency_ms: float) -> None:
        """
        Append a latency sample to the buffer.

        Args:
            latency_ms: Measured latency in milliseconds.
        """
        with self._lock:
            self._buf.append(latency_ms)

    def percentile(self, p: float) -> Optional[float]:
        """
        Compute the p-th percentile using linear interpolation.

        Algorithm (Nearest Rank / Linear Interpolation):
          1. Sort a copy of the buffer.
          2. Compute the fractional index: idx = (p / 100) * (n - 1)
          3. Interpolate between the two surrounding values.

        This is the same method used by numpy.percentile with
        interpolation='linear', giving smooth results even with small samples.

        Args:
            p: Percentile as a value between 0 and 100.
               e.g. p=50 → median, p=95 → p95, p=99 → p99

        Returns:
            Percentile value in milliseconds, or None if the buffer is empty.
        """
        with self._lock:
            if not self._buf:
                return None
            # Sort a list copy — we can't sort the deque in-place.
            sorted_vals = sorted(self._buf)

        n = len(sorted_vals)

        # Fractional index within the sorted array.
        # p=50 on 100 samples → idx = 49.5 → interpolate between [49] and [50].
        idx = (p / 100.0) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)    # clamp to valid index
        fraction = idx - lower            # how far between lower and upper

        # Linear interpolation between the two bounding values.
        return sorted_vals[lower] * (1.0 - fraction) + sorted_vals[upper] * fraction

    @property
    def count(self) -> int:
        """Return the current number of samples in the buffer."""
        # deque.__len__ is O(1) and thread-safe for reads on CPython.
        return len(self._buf)


# ---------------------------------------------------------------------------
# Per-provider stats
# ---------------------------------------------------------------------------

@dataclass
class ProviderStats:
    """
    Counters and a latency buffer for a single upstream provider.

    Attributes:
        requests:   Total requests forwarded to this provider.
        errors:     Requests that resulted in a provider-side error.
        latency:    Latency buffer for computing percentiles.
    """
    requests: int = 0
    errors: int = 0
    latency: LatencyBuffer = field(default_factory=LatencyBuffer)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Central in-memory metrics store.

    Intended to be instantiated once at startup and shared across the entire
    application (injected via FastAPI dependency or stored on app.state).

    All public methods are thread-safe.

    Usage::

        metrics = MetricsCollector()

        # In the request handler:
        metrics.record_request(provider="openai", model="gpt-4o", latency_ms=320.5)
        metrics.record_cache_hit()

        # In the /metrics endpoint:
        return metrics.snapshot()
    """

    def __init__(self) -> None:
        """
        Initialise all counters and buffers to zero.

        We use a single threading.Lock to protect all writes.  Reads in
        snapshot() also acquire the lock to prevent reading a partially-updated
        state (e.g. total_requests incremented but latency not yet added).
        """
        self._lock = threading.Lock()
        self._init_counters()

    def _init_counters(self) -> None:
        """
        Set (or reset) all counters and buffers to initial values.

        Extracted to a method so reset() can reuse it without duplicating code.
        """
        # ---- Global request counters ----
        self.total_requests: int = 0       # every proxied request (inc. errors)
        self.cache_hits: int = 0           # responses served from semantic cache
        self.cache_misses: int = 0         # requests that had to go to the provider
        self.rate_limited: int = 0         # requests rejected by rate limiter (429)
        self.contract_violations: int = 0  # responses blocked by contract evaluator

        # ---- Security counters ----
        self.injections_blocked_tier1: int = 0
        self.injections_blocked_tier2: int = 0

        # ---- Contract counters ----
        self.contract_blocks: int = 0
        self.contract_flags: int = 0
        self.contract_eval_latency: LatencyBuffer = LatencyBuffer(maxlen=1000)
        self.contract_tier_counts: dict[str, int] = {
            "deterministic": 0,
            "classifier": 0,
            "llm_judge": 0,
        }
        self.drift_alerts_total: int = 0

        # ---- Global latency buffer ----
        self.latency: LatencyBuffer = LatencyBuffer(maxlen=2000)

        # ---- Per-provider stats ----
        self.providers: dict[str, ProviderStats] = {}

    # -----------------------------------------------------------------------
    # Record methods — called by main.py after each pipeline stage
    # -----------------------------------------------------------------------

    def record_request(
        self,
        provider: str,
        model: str,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """
        Record a completed proxy request (success or provider error).

        Updates:
          - total_requests (global)
          - latency buffer (global)
          - providers[provider].requests and .latency (per-provider)
          - providers[provider].errors if error=True

        Args:
            provider:   Provider name ("openai", "anthropic", "local").
            model:      Model alias or canonical model ID used.
            latency_ms: End-to-end latency from request receipt to response.
            error:      True if the provider returned an error response.
        """
        with self._lock:
            self.total_requests += 1
            self.latency.add(latency_ms)

            # Ensure the per-provider ProviderStats entry exists.
            if provider not in self.providers:
                self.providers[provider] = ProviderStats()

            self.providers[provider].requests += 1
            self.providers[provider].latency.add(latency_ms)

            if error:
                self.providers[provider].errors += 1

    def record_cache_hit(self) -> None:
        """Increment the semantic cache hit counter."""
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Increment the semantic cache miss counter."""
        with self._lock:
            self.cache_misses += 1

    def record_injection_blocked(self, tier: int) -> None:
        """
        Increment the injection block counter for the given tier.

        Args:
            tier: 1 = PatternGuard blocked it, 2 = MLGuard blocked it.
        """
        with self._lock:
            if tier == 1:
                self.injections_blocked_tier1 += 1
            elif tier == 2:
                self.injections_blocked_tier2 += 1

    def record_rate_limited(self) -> None:
        """Increment the rate-limited request counter."""
        with self._lock:
            self.rate_limited += 1

    def record_contract_violation(self) -> None:
        """Increment the behavioral contract violation counter (BLOCK action)."""
        with self._lock:
            self.contract_violations += 1
            self.contract_blocks += 1

    def record_contract_flag(self) -> None:
        """Increment the contract flag counter (background FLAG action)."""
        with self._lock:
            self.contract_flags += 1

    def record_contract_evaluation(self, tier: str, latency_ms: float) -> None:
        """Record a contract evaluation with its tier and latency."""
        with self._lock:
            if tier in self.contract_tier_counts:
                self.contract_tier_counts[tier] += 1
            self.contract_eval_latency.add(latency_ms)

    def record_drift_alert(self) -> None:
        """Increment the drift alert counter."""
        with self._lock:
            self.drift_alerts_total += 1

    def record_provider_error(self, provider: str) -> None:
        """
        Increment the error counter for a specific provider.

        Creates the ProviderStats entry if it doesn't exist yet.

        Args:
            provider: The provider name that returned an error.
        """
        with self._lock:
            if provider not in self.providers:
                self.providers[provider] = ProviderStats()
            self.providers[provider].errors += 1

    # -----------------------------------------------------------------------
    # Snapshot — called by GET /metrics
    # -----------------------------------------------------------------------

    def snapshot(self) -> dict:
        """
        Return a point-in-time snapshot of all metrics as a JSON-safe dict.

        We acquire the lock for the entire snapshot to ensure all numbers are
        consistent with each other (e.g. hit_rate is computed from the same
        hits and misses that appear in the response).

        All Optional[float] values (percentiles) may be None if there are no
        latency samples yet — callers should handle this.

        Returns:
            A dict safe to pass directly to JSONResponse:

            {
              "requests": {
                "total": 1042,
                "rate_limited": 5,
                "contract_violations": 2
              },
              "cache": {
                "hits": 312,
                "misses": 730,
                "hit_rate": 0.299
              },
              "security": {
                "tier1_blocks": 3,
                "tier2_blocks": 1,
                "total_blocks": 4
              },
              "latency_ms": {
                "p50": 210.4,
                "p95": 580.2,
                "p99": 940.1,
                "samples": 1042
              },
              "providers": {
                "openai": {
                  "requests": 800,
                  "errors": 2,
                  "error_rate": 0.0025,
                  "p50_ms": 285.0,
                  "p95_ms": 520.3
                }
              }
            }
        """
        with self._lock:
            total = self.total_requests
            hits = self.cache_hits
            misses = self.cache_misses
            rate_limited = self.rate_limited
            violations = self.contract_violations
            t1_blocks = self.injections_blocked_tier1
            t2_blocks = self.injections_blocked_tier2

            c_blocks = self.contract_blocks
            c_flags = self.contract_flags
            c_tier_counts = dict(self.contract_tier_counts)
            c_drift_alerts = self.drift_alerts_total

            provider_snapshot = {
                name: {
                    "requests": stats.requests,
                    "errors": stats.errors,
                }
                for name, stats in self.providers.items()
            }

            latency_samples = self.latency.count

        # Compute derived values OUTSIDE the lock (percentile sorting is slow).
        # These are computed from the buffer's own internal lock, so they're safe.
        cache_total = hits + misses
        hit_rate = hits / cache_total if cache_total > 0 else 0.0

        provider_details = {}
        for name, counts in provider_snapshot.items():
            req = counts["requests"]
            err = counts["errors"]
            provider_details[name] = {
                "requests": req,
                "errors": err,
                "error_rate": round(err / req, 4) if req > 0 else 0.0,
                "p50_ms": self.providers[name].latency.percentile(50),
                "p95_ms": self.providers[name].latency.percentile(95),
            }

        return {
            "requests": {
                "total": total,
                "rate_limited": rate_limited,
                "contract_violations": violations,
            },
            "cache": {
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hit_rate, 4),
            },
            "security": {
                "tier1_blocks": t1_blocks,
                "tier2_blocks": t2_blocks,
                "total_blocks": t1_blocks + t2_blocks,
            },
            "contracts": {
                "blocks": c_blocks,
                "flags": c_flags,
                "total_violations": violations,
                "drift_alerts": c_drift_alerts,
                "evaluations_by_tier": c_tier_counts,
                "eval_latency_p50_ms": self.contract_eval_latency.percentile(50),
                "eval_latency_p95_ms": self.contract_eval_latency.percentile(95),
            },
            "latency_ms": {
                "p50": self.latency.percentile(50),
                "p95": self.latency.percentile(95),
                "p99": self.latency.percentile(99),
                "samples": latency_samples,
            },
            "providers": provider_details,
        }

    def reset(self) -> None:
        """
        Reset all counters and buffers to zero.

        Used in test suites to ensure clean state between test cases.
        Also useful for an admin "reset metrics" endpoint.

        Note: This creates NEW LatencyBuffer objects — old buffer references
        held by callers will not be reset.  This is acceptable in practice
        since buffers are only accessed through this class.
        """
        with self._lock:
            self._init_counters()
