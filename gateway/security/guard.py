"""
gateway/security/guard.py

Orchestrates the two-tier prompt injection detection pipeline and exposes a
single, clean `scan()` interface to the rest of the gateway.

Responsibilities:
  - Instantiate and own both PatternGuard (Tier 1) and MLGuard (Tier 2).
  - Implement the short-circuit logic: if Tier 1 detects an injection, skip
    the more expensive Tier 2 ML inference entirely.
  - Merge results from both tiers into a unified GuardResult dataclass.
  - Apply configurable severity policies:
      * Pattern severity < threshold → run ML as second opinion.
      * Pattern severity >= threshold → block immediately, skip ML.
  - Support "dry run" mode where injections are logged but not blocked (useful
    during classifier evaluation and policy tuning).
  - Expose the warm_up() method that delegates to MLGuard.warm_up() for use in
    the FastAPI lifespan hook.

Key classes / functions:
  - GuardResult              — unified result from the full scan pipeline
  - SecurityGuard            — main orchestrator class
    - __init__(settings)     — instantiate both guards with settings
    - warm_up()              — async: warm up the ML model
    - scan(text)             — async: run full pipeline, return GuardResult
    - _should_run_ml()       — internal: decide whether to invoke Tier 2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gateway.config import SecuritySettings
from gateway.security.pattern_guard import PatternGuard, PatternMatch, PatternSeverity
from gateway.security.ml_guard import MLGuard, ScanResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GuardResult:
    """
    Unified output from a full SecurityGuard.scan() call.

    Attributes:
        blocked:          True if the request should be rejected.
        reason:           Human-readable explanation (empty string if clean).
        tier_triggered:   Which tier made the blocking decision: 1, 2, or None.
        pattern_match:    PatternMatch from Tier 1 (None if no pattern triggered).
        ml_result:        ScanResult from Tier 2 (None if ML was skipped).
        total_latency_ms: Wall-clock time for the combined scan.
    """
    blocked: bool
    reason: str
    tier_triggered: Optional[int]
    pattern_match: Optional[PatternMatch]
    ml_result: Optional[ScanResult]
    total_latency_ms: float


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class SecurityGuard:
    """
    Two-tier injection detection orchestrator.

    Tier 1 (PatternGuard) always runs first because it is extremely fast.
    Tier 2 (MLGuard) runs only when:
      a) Tier 1 found nothing, OR
      b) Tier 1 found a LOW/MEDIUM severity match (and we want ML confirmation).

    This design keeps the common case (clean prompt) at < 2 ms end-to-end
    while still catching sophisticated injections that evade simple patterns.

    Usage::

        guard = SecurityGuard(settings.security)
        await guard.warm_up()

        result = await guard.scan(user_prompt)
        if result.blocked:
            raise HTTPException(400, detail=result.reason)
    """

    # Pattern severity levels at or above this value trigger an immediate block
    # without waiting for the ML model.
    _IMMEDIATE_BLOCK_SEVERITY = PatternSeverity.HIGH

    def __init__(self, settings: SecuritySettings) -> None:
        """
        Initialise both guard tiers.

        Args:
            settings: SecuritySettings containing paths, thresholds, and
                      enabled flags for both tiers.
        """
        self._settings = settings
        self._pattern_guard = PatternGuard(settings.pattern_file_path)
        self._ml_guard = MLGuard(settings)
        self._dry_run: bool = False  # set via environment / admin API

    async def warm_up(self) -> None:
        """
        Warm up the ML model by delegating to MLGuard.warm_up().

        Should be called from the FastAPI lifespan startup hook.  Safe to call
        multiple times (the underlying load is idempotent).
        """
        # TODO: delegate to self._ml_guard.warm_up() if ml_guard_enabled
        ...

    async def scan(self, text: str) -> GuardResult:
        """
        Run the full two-tier injection scan pipeline.

        Pipeline:
          1. Run PatternGuard.scan() (always, unless pattern_guard_enabled=False).
          2. If a HIGH/CRITICAL pattern fires → block immediately.
          3. Else if ML guard is enabled → run MLGuard.scan().
          4. Merge results into a GuardResult.
          5. If dry_run mode is on → set blocked=False regardless of findings.

        Args:
            text: The prompt string to analyse.

        Returns:
            GuardResult with the combined decision and all intermediate data.
        """
        # TODO: implement
        ...

    def _should_run_ml(self, pattern_match: Optional[PatternMatch]) -> bool:
        """
        Decide whether Tier 2 ML inference is needed given Tier 1's output.

        Logic:
          - If pattern_guard_enabled is False → always run ML (if enabled).
          - If no pattern match → run ML.
          - If pattern severity is LOW or MEDIUM → run ML for second opinion.
          - If pattern severity is HIGH or CRITICAL → skip ML (already blocking).

        Args:
            pattern_match: The result from PatternGuard.scan(), or None.

        Returns:
            True if MLGuard.scan() should be invoked.
        """
        # TODO: implement
        ...
