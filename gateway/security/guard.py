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

    DESIGN RATIONALE — WHY TWO TIERS?

    Tier 1 (PatternGuard) is a regex scanner:
      + Extremely fast: < 1 ms per prompt
      + Zero false positives on explicitly-listed patterns
      + No GPU or model loading required
      - Can only catch known patterns; creative attackers evade it

    Tier 2 (MLGuard) is a DistilBERT classifier:
      + Generalises to paraphrased, novel, or obfuscated injections
      + Learns statistical patterns across thousands of training examples
      - Slower: 30-100 ms on CPU
      - Requires a trained model on disk

    The two-tier pipeline gets the best of both:
      - Common, obvious injections are caught instantly by Tier 1.
      - Sophisticated attacks that evade Tier 1 are caught by Tier 2.
      - Clean prompts pay only the Tier 1 cost (< 1 ms), then Tier 2 (30-100 ms).
      - HIGH/CRITICAL pattern matches skip Tier 2 entirely (already blocking).

    Usage::

        guard = SecurityGuard(settings.security)
        await guard.warm_up()

        result = await guard.scan(user_prompt)
        if result.blocked:
            raise HTTPException(400, detail=result.reason)
    """

    # Pattern severities that cause an immediate block WITHOUT waiting for ML.
    # HIGH and CRITICAL patterns are high-confidence matches — running the ML
    # model for a "second opinion" would just add latency with no benefit.
    _IMMEDIATE_BLOCK_SEVERITIES = {PatternSeverity.HIGH, PatternSeverity.CRITICAL}

    def __init__(self, settings: SecuritySettings) -> None:
        """
        Initialise both guard tiers from settings.

        Args:
            settings: SecuritySettings containing paths, thresholds, and
                      enabled flags for both tiers.
        """
        self._settings = settings

        # Tier 1: created immediately (regex compilation is cheap).
        # If pattern_guard_enabled=False, we still create the object but
        # skip calling .scan() on it in scan().
        self._pattern_guard = PatternGuard(settings.pattern_file_path)

        # Tier 2: created but NOT loaded yet.  warm_up() loads the model.
        self._ml_guard = MLGuard(settings)

        # Dry-run mode: detect and log injections but never block.
        # Useful when deploying the security guard to a new environment and
        # you want to observe what it would block before enabling enforcement.
        self._dry_run: bool = False

    async def warm_up(self) -> None:
        """
        Warm up the ML model by delegating to MLGuard.warm_up().

        Idempotent — safe to call multiple times.  Each extra call after the
        first is a no-op because MLGuard tracks whether loading was attempted.

        Called from the FastAPI lifespan startup hook.
        """
        if self._settings.ml_guard_enabled:
            await self._ml_guard.warm_up()

    async def scan(self, text: str) -> GuardResult:
        """
        Run the full two-tier injection scan and return a unified result.

        Pipeline (left to right, earliest exit wins):

          [Tier 1: pattern scan]
                │
                ├── Pattern severity HIGH/CRITICAL?  → BLOCK immediately (skip ML)
                │
                ├── Pattern severity LOW/MEDIUM?  → run ML for second opinion
                │       └── ML says injection?   → BLOCK (tier=2)
                │           ML says benign?      → PASS (low-severity pattern logged)
                │
                └── No pattern match?  → run ML
                        └── ML says injection? → BLOCK (tier=2)
                            ML says benign?    → PASS (clean prompt)

        Dry-run mode:
          In dry_run mode the pipeline runs fully but blocked is forced to False.
          This lets operators evaluate what the guard WOULD block in production
          before enabling enforcement.

        Args:
            text: Full conversation as a single string.

        Returns:
            GuardResult with blocked flag, reason, tier info, and raw sub-results.
        """
        import time
        t0 = time.perf_counter()

        pattern_match: Optional[PatternMatch] = None
        ml_result: Optional[ScanResult] = None
        blocked = False
        reason = ""
        tier_triggered: Optional[int] = None

        # ---------------------------------------------------------------
        # Tier 1: Pattern scan
        # ---------------------------------------------------------------
        if self._settings.pattern_guard_enabled:
            pattern_match = self._pattern_guard.scan(text)

            if pattern_match is not None:
                severity = pattern_match.severity

                if severity in self._IMMEDIATE_BLOCK_SEVERITIES:
                    # High confidence match — block without calling the ML model.
                    blocked = True
                    tier_triggered = 1
                    reason = (
                        f"Prompt injection detected by pattern guard "
                        f"(pattern: '{pattern_match.pattern_id}', "
                        f"severity: {severity.value})"
                    )
                # For LOW/MEDIUM severity we fall through to Tier 2 below.

        # ---------------------------------------------------------------
        # Tier 2: ML scan (only if not already blocking from Tier 1)
        # ---------------------------------------------------------------
        if not blocked and self._should_run_ml(pattern_match):
            ml_result = await self._ml_guard.scan(text)

            if ml_result.is_injection:
                blocked = True
                tier_triggered = 2
                reason = (
                    f"Prompt injection detected by ML classifier "
                    f"(confidence: {ml_result.confidence:.2%}, "
                    f"threshold: {self._settings.ml_confidence_threshold:.2%})"
                )

            # Edge case: LOW/MEDIUM pattern matched but ML says benign.
            # We trust the ML model here and pass the request through.
            # The pattern match is still logged via the returned GuardResult.

        # ---------------------------------------------------------------
        # Dry-run override
        # ---------------------------------------------------------------
        if self._dry_run and blocked:
            # We detected an injection but won't block in dry-run mode.
            # The GuardResult still carries all the detection info for logging.
            reason = f"[DRY RUN — would have blocked] {reason}"
            blocked = False

        total_ms = (time.perf_counter() - t0) * 1000

        return GuardResult(
            blocked=blocked,
            reason=reason,
            tier_triggered=tier_triggered,
            pattern_match=pattern_match,
            ml_result=ml_result,
            total_latency_ms=total_ms,
        )

    def _should_run_ml(self, pattern_match: Optional[PatternMatch]) -> bool:
        """
        Decide whether Tier 2 ML inference is needed.

        The ML guard is skipped if:
          a) ml_guard_enabled is False in settings.
          b) Tier 1 found a HIGH/CRITICAL pattern (we're already blocking,
             no point paying the ML inference cost).

        The ML guard runs if:
          a) No pattern matched at all (most common case for clean prompts).
          b) Tier 1 found a LOW/MEDIUM pattern (ML confirms or overrides).

        Args:
            pattern_match: Output of PatternGuard.scan(), or None if
                           pattern_guard_enabled is False.

        Returns:
            True if MLGuard.scan() should be invoked.
        """
        if not self._settings.ml_guard_enabled:
            return False

        if pattern_match is None:
            # No pattern matched — ML is our only safety net.
            return True

        # Pattern matched — run ML only for LOW/MEDIUM severity (not conclusive
        # enough to block without a second opinion).
        return pattern_match.severity not in self._IMMEDIATE_BLOCK_SEVERITIES
