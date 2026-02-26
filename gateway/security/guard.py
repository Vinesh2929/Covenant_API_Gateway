"""
gateway/security/guard.py

Orchestrates the three-tier prompt injection detection pipeline and exposes a
single, clean `scan()` interface to the rest of the gateway.

Responsibilities:
  - Instantiate and own PatternGuard (Tier 1), MLGuard (Tier 2), and
    optionally LLMGuard (Tier 3).
  - Implement the short-circuit logic: if Tier 1 detects an injection, skip
    the more expensive Tier 2 ML inference entirely.
  - Merge results from both tiers into a unified GuardResult dataclass.
  - Apply configurable severity policies:
      * Pattern severity < threshold → run ML as second opinion.
      * Pattern severity >= threshold → block immediately, skip ML.
  - Fire Tier 3 as a background asyncio task (fire-and-forget) when Tier 2
    scores fall in the ambiguous zone [llm_guard_low_threshold, llm_guard_high_threshold].
    Tier 3 is purely observational — it logs results but never blocks.
  - Support "dry run" mode where injections are logged but not blocked (useful
    during classifier evaluation and policy tuning).
  - Expose the warm_up() method that delegates to MLGuard.warm_up() for use in
    the FastAPI lifespan hook.

Key classes / functions:
  - GuardResult              — unified result from the full scan pipeline
  - SecurityGuard            — main orchestrator class
    - __init__(settings)     — instantiate all guard tiers with settings
    - warm_up()              — async: warm up the ML model
    - scan(text)             — async: run full pipeline, return GuardResult
    - _should_run_ml()       — internal: decide whether to invoke Tier 2
    - _run_llm_judge()       — internal async: Tier 3 background task
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import structlog

from gateway.config import SecuritySettings
from gateway.security.pattern_guard import PatternGuard, PatternMatch, PatternSeverity
from gateway.security.ml_guard import MLGuard, ScanResult
from gateway.security.llm_guard import LLMGuard, LLMJudgeResult

logger = structlog.get_logger(__name__)


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
        llm_result:       LLMJudgeResult from Tier 3 (None — always None at return
                          time because Tier 3 runs asynchronously in the background).
                          Present only in tests that directly await _run_llm_judge().
        total_latency_ms: Wall-clock time for Tier 1 + Tier 2 (excludes Tier 3,
                          which runs after the response is returned to the caller).
    """
    blocked: bool
    reason: str
    tier_triggered: Optional[int]
    pattern_match: Optional[PatternMatch]
    ml_result: Optional[ScanResult]
    total_latency_ms: float
    llm_result: Optional[LLMJudgeResult] = field(default=None)


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

    def __init__(self, settings: SecuritySettings, providers_settings=None) -> None:
        """
        Initialise all guard tiers from settings.

        Args:
            settings:          SecuritySettings containing paths, thresholds, and
                               enabled flags for all tiers.
            providers_settings: Optional ProviderSettings forwarded to LLMGuard
                               for API key resolution. When None, LLMGuard reads
                               keys from environment variables directly.
        """
        self._settings = settings

        # Tier 1: created immediately (regex compilation is cheap).
        # If pattern_guard_enabled=False, we still create the object but
        # skip calling .scan() on it in scan().
        self._pattern_guard = PatternGuard(settings.pattern_file_path)

        # Tier 2: created but NOT loaded yet.  warm_up() loads the model.
        self._ml_guard = MLGuard(settings)

        # Tier 3: async LLM judge — only instantiated when enabled.
        # Fires as a background task for ambiguous Tier-2 scores.
        self._llm_guard: Optional[LLMGuard] = None
        if settings.llm_guard_enabled:
            self._llm_guard = LLMGuard(settings, providers_settings)

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
        # Tier 3: async LLM judge (fire-and-forget, observational only)
        # ---------------------------------------------------------------
        # Triggered when:
        #   - LLMGuard is instantiated (llm_guard_enabled=True)
        #   - Tier 2 ran and the score falls in the ambiguous zone
        #   - The request is NOT already blocked (no point judging confirmed blocks)
        # The task runs in the background; scan() returns before it completes.
        if (
            self._llm_guard is not None
            and ml_result is not None
            and not blocked
            and self._settings.llm_guard_low_threshold
                <= ml_result.confidence
                <= self._settings.llm_guard_high_threshold
        ):
            asyncio.create_task(
                self._run_llm_judge(text, ml_result.confidence),
                name="tier3_judge",
            )

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

    async def _run_llm_judge(self, text: str, tier2_score: float) -> None:
        """
        Background Tier 3 task — call the LLM judge and log the result.

        This method is called via asyncio.create_task() from scan() and runs
        after scan() has already returned a result to the caller.  It is purely
        observational: it never modifies the GuardResult or blocks any request.

        Log levels:
          INFO    — judge says benign (or error)
          WARNING — judge says injection (disagreement with Tier 2)

        Args:
            text:        The prompt that was scanned.
            tier2_score: The Tier 2 ML confidence score (for context in logs).
        """
        result = await self._llm_guard.judge(text)  # type: ignore[union-attr]

        if result.error:
            logger.info(
                "tier3_judge_failed",
                tier2_score=round(tier2_score, 4),
                error=result.error,
                latency_ms=round(result.latency_ms, 1),
            )
            return

        if result.is_injection:
            # Tier 2 said benign (score below threshold), Tier 3 disagrees —
            # log at WARNING so these samples surface for training data review.
            logger.warning(
                "tier3_disagrees_with_tier2",
                tier2_score=round(tier2_score, 4),
                tier3_verdict="injection",
                reason=result.reason,
                model=result.model,
                latency_ms=round(result.latency_ms, 1),
            )
        else:
            logger.info(
                "tier3_agrees_benign",
                tier2_score=round(tier2_score, 4),
                tier3_verdict="benign",
                reason=result.reason,
                model=result.model,
                latency_ms=round(result.latency_ms, 1),
            )
