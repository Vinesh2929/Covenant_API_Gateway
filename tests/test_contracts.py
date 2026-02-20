"""
tests/test_contracts.py

Unit tests for the behavioral contract system (schema, evaluator, registry).

Test coverage goals:
  Schema / Pydantic models:
    - Valid contract definitions parse without error.
    - Invalid regex in RegexContract raises a validation error.
    - ContractDefinition with nested CompositeContract round-trips to/from JSON.

  Evaluators:
    KeywordEvaluator:
      - Blocked when response contains a keyword (case-insensitive).
      - Passes when response contains no keywords.
      - Respects case_sensitive=True.
    RegexEvaluator:
      - Blocked when response matches the pattern.
      - Passes when pattern does not match.
    SentimentEvaluator (stub):
      - Blocked when stub sentiment < min_score.
      - Passes when stub sentiment >= min_score.
    SchemaEvaluator:
      - Passes when response content is valid JSON conforming to schema.
      - Blocked when response content violates the schema.
    CompositeEvaluator:
      - AND: blocked only when all sub-contracts are violated.
      - OR:  blocked when any sub-contract is violated.

  ContractRegistry:
    - load() populates the registry from the fixture contracts directory.
    - evaluate() returns an unblocked report when no contracts are registered.
    - evaluate() returns a blocked report when a BLOCK-action contract is violated.
    - evaluate() returns a warn (not blocked) report for WARN-action violations.
    - register() and get() work programmatically.
    - reload() hot-reloads a single app's contracts.
"""

from __future__ import annotations

import pytest

from gateway.contracts.evaluator import (
    EvaluationResult,
    KeywordEvaluator,
    RegexEvaluator,
    SchemaEvaluator,
    evaluate_contract,
)
from gateway.contracts.registry import ContractRegistry, ContractReport
from gateway.contracts.schema import (
    ContractAction,
    ContractDefinition,
    KeywordContract,
    RegexContract,
    SchemaContract,
    SentimentContract,
    CompositeContract,
)


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------

class TestContractSchema:
    """Pydantic model parsing and validation."""

    def test_keyword_contract_parses(self):
        """A valid KeywordContract dict should parse without error."""
        # TODO: implement
        ...

    def test_invalid_regex_raises_validation_error(self):
        """An invalid regex pattern should raise a Pydantic ValidationError."""
        # TODO: implement
        ...

    def test_composite_contract_roundtrip(self):
        """A CompositeContract should serialise and deserialise correctly."""
        # TODO: implement — model_dump() → model_validate()
        ...

    def test_contract_definition_defaults(self):
        """ContractDefinition should have sensible defaults for optional fields."""
        # TODO: implement
        ...


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

class TestKeywordEvaluator:

    @pytest.mark.asyncio
    async def test_blocks_when_keyword_present(self, sample_request, sample_response):
        """Response containing a blocked keyword should fail the evaluation."""
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_passes_when_no_keyword_present(self, sample_request, sample_response):
        """Clean response should pass the keyword evaluation."""
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_case_insensitive_by_default(self, sample_request, sample_response):
        """Keyword match should be case-insensitive by default."""
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_case_sensitive_flag_respected(self, sample_request, sample_response):
        """With case_sensitive=True, mismatched case should not trigger."""
        # TODO: implement
        ...


class TestRegexEvaluator:

    @pytest.mark.asyncio
    async def test_blocks_when_pattern_matches(self, sample_request, sample_response):
        """Response matching the regex should fail the evaluation."""
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_passes_when_pattern_does_not_match(
        self, sample_request, sample_response
    ):
        """Response not matching the regex should pass."""
        # TODO: implement
        ...


class TestSchemaEvaluator:

    @pytest.mark.asyncio
    async def test_passes_valid_json_content(self, sample_request):
        """Response content conforming to the JSON Schema should pass."""
        # TODO: implement — response with structured JSON content
        ...

    @pytest.mark.asyncio
    async def test_blocks_invalid_json_content(self, sample_request):
        """Response content violating the JSON Schema should be blocked."""
        # TODO: implement
        ...


class TestCompositeEvaluator:

    @pytest.mark.asyncio
    async def test_and_blocks_only_when_all_violated(
        self, sample_request, sample_response
    ):
        """AND composite: should only block if EVERY sub-contract is violated."""
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_or_blocks_when_any_violated(self, sample_request, sample_response):
        """OR composite: should block if ANY sub-contract is violated."""
        # TODO: implement
        ...


# ---------------------------------------------------------------------------
# ContractRegistry tests
# ---------------------------------------------------------------------------

class TestContractRegistry:

    def test_load_populates_registry(self, contract_registry):
        """After load(), the registry should contain the test app contracts."""
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_evaluate_no_contracts_returns_unblocked(
        self, sample_request, sample_response
    ):
        """An app with no registered contracts should always pass."""
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_evaluate_block_action_sets_blocked(
        self, contract_registry, sample_request, sample_response
    ):
        """A violated BLOCK-action contract should set report.blocked=True."""
        # TODO: implement
        ...

    @pytest.mark.asyncio
    async def test_evaluate_warn_action_does_not_block(
        self, contract_registry, sample_request, sample_response
    ):
        """A violated WARN-action contract should set report.blocked=False."""
        # TODO: implement
        ...

    def test_programmatic_register_and_get(self):
        """register() + get() should work for dynamically added contracts."""
        # TODO: implement
        ...

    def test_reload_updates_single_app_contracts(self, contract_registry, tmp_path):
        """reload(app_id) should re-read the file for that app only."""
        # TODO: implement
        ...
