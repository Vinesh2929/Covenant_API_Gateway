"""
tests/test_contracts.py — tests for the behavioral contracts system.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gateway.contracts.schema import (
    ContractAction,
    ContractDefinition,
    CompositeContract,
    KeywordContract,
    LengthLimitContract,
    RegexContract,
    SchemaContract,
)
from gateway.contracts.evaluator import (
    EvaluationResult,
    KeywordEvaluator,
    LengthLimitEvaluator,
    RegexEvaluator,
    SchemaEvaluator,
    CompositeEvaluator,
    evaluate_contract,
)
from gateway.contracts.registry import ContractRegistry


def _make_response(content: str) -> dict:
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


REQ = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestContractSchema:

    def test_keyword_contract_parses(self):
        c = KeywordContract(
            id="kw-1", name="No Swearing", keywords=["damn", "hell"],
        )
        assert c.type.value == "keyword"
        assert len(c.keywords) == 2
        assert c.action == ContractAction.BLOCK

    def test_invalid_regex_raises_validation_error(self):
        with pytest.raises(ValidationError):
            RegexContract(id="bad-rx", name="Bad", pattern="[invalid")

    def test_composite_contract_roundtrip(self):
        sub = KeywordContract(id="sub-1", name="Sub", keywords=["test"])
        comp = CompositeContract(
            id="comp-1", name="Composite", operator="OR", contracts=[sub],
        )
        data = comp.model_dump(mode="json")
        restored = CompositeContract.model_validate(data)
        assert restored.id == "comp-1"
        assert len(restored.contracts) == 1

    def test_contract_definition_defaults(self):
        sub = KeywordContract(id="k1", name="K", keywords=["x"])
        defn = ContractDefinition(app_id="my-app", contracts=[sub])
        assert defn.version == "1.0.0"
        assert defn.app_id == "my-app"


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

class TestKeywordEvaluator:

    @pytest.mark.asyncio
    async def test_blocks_when_keyword_present(self):
        contract = KeywordContract(
            id="kw-block", name="Block Bad", keywords=["forbidden"],
            action=ContractAction.BLOCK,
        )
        result = await evaluate_contract(contract, REQ, _make_response("This is forbidden content"))
        assert result.passed is False
        assert result.compliance_score == 0.0

    @pytest.mark.asyncio
    async def test_passes_when_no_keyword(self):
        contract = KeywordContract(
            id="kw-pass", name="Clean", keywords=["forbidden"],
        )
        result = await evaluate_contract(contract, REQ, _make_response("This is perfectly fine"))
        assert result.passed is True
        assert result.compliance_score == 1.0

    @pytest.mark.asyncio
    async def test_case_insensitive_by_default(self):
        contract = KeywordContract(
            id="kw-ci", name="CI", keywords=["HELLO"],
        )
        result = await evaluate_contract(contract, REQ, _make_response("hello world"))
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_case_sensitive_respected(self):
        contract = KeywordContract(
            id="kw-cs", name="CS", keywords=["Hello"], case_sensitive=True,
        )
        result = await evaluate_contract(contract, REQ, _make_response("hello world"))
        assert result.passed is True


class TestRegexEvaluator:

    @pytest.mark.asyncio
    async def test_blocks_on_match(self):
        contract = RegexContract(
            id="rx-block", name="SSN", pattern=r"\d{3}-\d{2}-\d{4}",
        )
        result = await evaluate_contract(contract, REQ, _make_response("SSN: 123-45-6789"))
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_passes_no_match(self):
        contract = RegexContract(
            id="rx-pass", name="SSN", pattern=r"\d{3}-\d{2}-\d{4}",
        )
        result = await evaluate_contract(contract, REQ, _make_response("No sensitive data here"))
        assert result.passed is True


class TestLengthLimitEvaluator:

    @pytest.mark.asyncio
    async def test_blocks_over_word_limit(self):
        contract = LengthLimitContract(
            id="len-block", name="Short", max_words=5,
        )
        result = await evaluate_contract(
            contract, REQ,
            _make_response("one two three four five six seven eight nine ten"),
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_passes_under_limit(self):
        contract = LengthLimitContract(
            id="len-pass", name="Short", max_words=10,
        )
        result = await evaluate_contract(contract, REQ, _make_response("hello world"))
        assert result.passed is True


class TestSchemaEvaluator:

    @pytest.mark.asyncio
    async def test_passes_valid_json(self):
        schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
        contract = SchemaContract(
            id="sch-pass", name="JSON", json_schema=schema,
            target_field="choices.0.message.content",
        )
        response = _make_response('{"name": "Alice"}')
        result = await evaluate_contract(contract, REQ, response)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_blocks_invalid_json(self):
        schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
        contract = SchemaContract(
            id="sch-block", name="JSON", json_schema=schema,
            target_field="choices.0.message.content",
        )
        response = _make_response('{"age": 30}')
        result = await evaluate_contract(contract, REQ, response)
        assert result.passed is False


class TestCompositeEvaluator:

    @pytest.mark.asyncio
    async def test_or_blocks_when_any_violated(self):
        passing = KeywordContract(id="c-pass", name="P", keywords=["xyz123"])
        failing = KeywordContract(id="c-fail", name="F", keywords=["hello"])
        comp = CompositeContract(
            id="comp-or", name="OR", operator="OR", contracts=[passing, failing],
        )
        result = await evaluate_contract(comp, REQ, _make_response("hello world"))
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_and_blocks_only_when_all_violated(self):
        passing = KeywordContract(id="c-pass2", name="P", keywords=["xyz123"])
        failing = KeywordContract(id="c-fail2", name="F", keywords=["hello"])
        comp = CompositeContract(
            id="comp-and", name="AND", operator="AND", contracts=[passing, failing],
        )
        result = await evaluate_contract(comp, REQ, _make_response("hello world"))
        # AND: only blocks when ALL fail. "passing" passes, so composite passes.
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_and_blocks_when_all_fail(self):
        fail1 = KeywordContract(id="f1", name="F1", keywords=["hello"])
        fail2 = KeywordContract(id="f2", name="F2", keywords=["world"])
        comp = CompositeContract(
            id="comp-and2", name="AND2", operator="AND", contracts=[fail1, fail2],
        )
        result = await evaluate_contract(comp, REQ, _make_response("hello world"))
        assert result.passed is False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestContractRegistry:

    @pytest.mark.asyncio
    async def test_evaluate_no_contracts_returns_unblocked(self):
        registry = ContractRegistry(contracts_dir="nonexistent_dir/")
        report = await registry.evaluate("unknown-app", REQ, _make_response("hi"))
        assert report.blocked is False
        assert report.violations == []

    def test_programmatic_register_and_get(self):
        registry = ContractRegistry(contracts_dir="contracts/")
        defn = ContractDefinition(
            app_id="test-app",
            contracts=[KeywordContract(id="t1", name="T", keywords=["bad"])],
        )
        registry.register(defn)
        assert registry.get("test-app") is not None
        assert "test-app" in registry.list_apps()

    @pytest.mark.asyncio
    async def test_block_contract_sets_blocked(self):
        registry = ContractRegistry(contracts_dir="contracts/")
        defn = ContractDefinition(
            app_id="block-test",
            contracts=[
                KeywordContract(
                    id="kw-blocker", name="Blocker",
                    keywords=["secret"], action=ContractAction.BLOCK,
                ),
            ],
        )
        registry.register(defn)
        report = await registry.evaluate("block-test", REQ, _make_response("This is a secret"))
        assert report.blocked is True
        assert len(report.violations) == 1

    @pytest.mark.asyncio
    async def test_flag_contract_does_not_block(self):
        registry = ContractRegistry(contracts_dir="contracts/")
        defn = ContractDefinition(
            app_id="flag-test",
            contracts=[
                KeywordContract(
                    id="kw-flagger", name="Flagger",
                    keywords=["secret"], action=ContractAction.FLAG,
                ),
            ],
        )
        registry.register(defn)
        report = await registry.evaluate("flag-test", REQ, _make_response("This is a secret"))
        # FLAG contracts run in background — the report should NOT be blocked
        assert report.blocked is False
