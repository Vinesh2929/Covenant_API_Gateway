"""
tests/test_security.py

Unit tests for the security package (pattern_guard, ml_guard, guard).

Test coverage goals:
  PatternGuard:
    - Clean prompts return None.
    - Known injection patterns return a PatternMatch with the correct ID and severity.
    - Pattern matching is case-insensitive (when flag is set).
    - Severity levels are respected in the returned PatternMatch.
    - reload() updates the pattern list without restarting.

  MLGuard (via stub):
    - A prompt flagged by the stub returns ScanResult(is_injection=True).
    - A clean prompt returns ScanResult(is_injection=False).
    - Confidence below the threshold is not blocked.
    - warm_up() is idempotent (can be called multiple times safely).

  SecurityGuard (orchestration):
    - HIGH-severity pattern triggers immediate block without calling ML.
    - LOW-severity pattern calls ML and defers to its result.
    - Clean prompt passes both tiers.
    - dry_run=True logs but does not block.
    - GuardResult.tier_triggered is set correctly for each case.
"""

from __future__ import annotations

import pytest

from gateway.security.guard import GuardResult, SecurityGuard
from gateway.security.ml_guard import ScanResult
from gateway.security.pattern_guard import PatternGuard, PatternMatch, PatternSeverity


# ---------------------------------------------------------------------------
# PatternGuard tests
# ---------------------------------------------------------------------------

class TestPatternGuard:
    """Unit tests for the regex-based Tier-1 guard."""

    def test_clean_prompt_returns_none(self, pattern_guard):
        """A benign prompt should produce no match."""
        # TODO: implement
        ...

    def test_known_jailbreak_returns_match(self, pattern_guard):
        """A classic 'ignore previous instructions' prompt should trigger."""
        # TODO: implement
        ...

    def test_match_contains_correct_severity(self, pattern_guard):
        """The returned PatternMatch should carry the pattern's defined severity."""
        # TODO: implement
        ...

    def test_case_insensitive_matching(self, pattern_guard):
        """Pattern matching should be case-insensitive for patterns that specify it."""
        # TODO: implement
        ...

    def test_reload_updates_patterns(self, pattern_guard, tmp_path):
        """
        Writing a new patterns file to disk and calling reload() should update
        the active pattern list.
        """
        # TODO: implement
        ...

    def test_list_patterns_returns_all_loaded(self, pattern_guard):
        """list_patterns() should return one dict per loaded pattern."""
        # TODO: implement
        ...


# ---------------------------------------------------------------------------
# MLGuard tests (via stub)
# ---------------------------------------------------------------------------

class TestMLGuard:
    """Unit tests for the DistilBERT Tier-2 guard (using stub model)."""

    @pytest.mark.asyncio
    async def test_injection_above_threshold_returns_blocked(self, ml_guard_stub):
        """When stub returns confidence > threshold, is_injection should be True."""
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_confidence_below_threshold_not_blocked(self, ml_guard_stub):
        """When stub returns confidence < threshold, is_injection should be False."""
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_warm_up_is_idempotent(self, ml_guard_stub):
        """Calling warm_up() multiple times should not raise."""
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_unloaded_model_returns_safe_result(self, settings):
        """
        If the model path does not exist, scan() should return a ScanResult
        with is_injection=False rather than raising an exception.
        """
        # TODO: implement
        ...


# ---------------------------------------------------------------------------
# SecurityGuard orchestration tests
# ---------------------------------------------------------------------------

class TestSecurityGuard:
    """Integration tests for the two-tier orchestration logic."""

    @pytest.mark.asyncio
    async def test_clean_prompt_passes_both_tiers(self, security_guard):
        """A benign prompt should result in GuardResult(blocked=False)."""
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_high_severity_pattern_blocks_immediately(
        self, security_guard, mocker
    ):
        """
        A HIGH-severity pattern match should block the request without calling
        the ML guard (verify ML scan is not invoked via mock).
        """
        # TODO: implement — mocker.spy(security_guard._ml_guard, "scan")
        ...

    @pytest.mark.asyncio
    async def test_low_severity_pattern_invokes_ml(self, security_guard, mocker):
        """
        A LOW-severity pattern match should trigger the ML guard for a second
        opinion.
        """
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_dry_run_does_not_block(self, security_guard):
        """
        In dry_run mode, even a HIGH-severity detection should return
        GuardResult(blocked=False).
        """
        # TODO: implement — set security_guard._dry_run = True
        ...

    @pytest.mark.asyncio
    async def test_tier_triggered_set_correctly(self, security_guard):
        """
        GuardResult.tier_triggered should be 1 when a pattern blocks,
        2 when only the ML guard blocks, and None when the request is clean.
        """
        # TODO: implement
        ...
