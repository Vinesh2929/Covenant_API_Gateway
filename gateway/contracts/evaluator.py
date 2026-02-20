"""
gateway/contracts/evaluator.py

Evaluates a single behavioral contract rule against a (request, response) pair.

Responsibilities:
  - Implement an evaluation function for each contract type (keyword, regex,
    sentiment, schema, composite).
  - Return a structured EvaluationResult that records: which contract fired,
    the violation details, and whether the request should be blocked or just
    logged.
  - Be stateless: evaluators receive all context they need as arguments and
    produce deterministic outputs (no side effects).
  - Make it easy to add new contract types by implementing the ContractEvaluator
    protocol — one evaluate() method per type.

Key classes / functions:
  - ViolationDetail           — dataclass: what exactly was violated and where
  - EvaluationResult          — dataclass: passed, action, contract_id, violation
  - ContractEvaluator         — abstract base / Protocol for per-type evaluators
  - KeywordEvaluator          — evaluates KeywordContract
  - RegexEvaluator            — evaluates RegexContract
  - SentimentEvaluator        — evaluates SentimentContract (async, uses HF model)
  - SchemaEvaluator           — evaluates SchemaContract (uses jsonschema library)
  - CompositeEvaluator        — evaluates CompositeContract recursively
  - evaluate_contract(...)    — top-level dispatcher function
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from gateway.contracts.schema import (
    AnyContract,
    CompositeContract,
    ContractAction,
    KeywordContract,
    RegexContract,
    SchemaContract,
    SentimentContract,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ViolationDetail:
    """
    Describes the specific aspect of the response that violated the contract.

    Attributes:
        contract_id:  The ID of the contract rule that was violated.
        message:      Human-readable description of the violation.
        evidence:     The offending text snippet or field value.
        location:     Where in the response the violation was found (e.g. field path).
    """
    contract_id: str
    message: str
    evidence: Optional[str] = None
    location: Optional[str] = None


@dataclass
class EvaluationResult:
    """
    Output of a single contract evaluation.

    Attributes:
        passed:      True if the response satisfies the contract.
        action:      The ContractAction to take if not passed (None if passed).
        contract_id: The ID of the evaluated contract.
        violation:   ViolationDetail if not passed, else None.
    """
    passed: bool
    action: Optional[ContractAction]
    contract_id: str
    violation: Optional[ViolationDetail] = None


# ---------------------------------------------------------------------------
# Evaluator protocol
# ---------------------------------------------------------------------------

class ContractEvaluator(Protocol):
    """
    Protocol that all per-type evaluators must satisfy.

    Each evaluator is responsible for exactly one AnyContract subtype and
    exposes a single evaluate() method.
    """

    async def evaluate(
        self,
        contract: AnyContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        """
        Evaluate `contract` against the given request/response pair.

        Args:
            contract: The contract definition (subtype must match the evaluator).
            request:  The original chat completion request body (normalised).
            response: The upstream provider response body (normalised).

        Returns:
            An EvaluationResult with passed=True or the violation details.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete evaluators
# ---------------------------------------------------------------------------

class KeywordEvaluator:
    """
    Evaluates a KeywordContract by scanning the response content text for
    any of the listed keywords.
    """

    async def evaluate(
        self,
        contract: KeywordContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        """
        Search the response message content for each keyword in the contract.

        Args:
            contract: A KeywordContract with the keywords list and flags.
            request:  Normalised request body (unused by this evaluator).
            response: Normalised response body; checks choices[0].message.content.

        Returns:
            EvaluationResult(passed=False) if any keyword is found, else passed=True.
        """
        # TODO: implement — extract response text, iterate keywords, check match
        ...


class RegexEvaluator:
    """
    Evaluates a RegexContract by running re.search on the response content.
    """

    async def evaluate(
        self,
        contract: RegexContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        """
        Compile the contract's regex (with any specified flags) and search
        the response content.

        Args:
            contract: A RegexContract with pattern and optional flags list.
            request:  Normalised request body.
            response: Normalised response body.

        Returns:
            EvaluationResult(passed=False) if the pattern matches, else passed=True.
        """
        # TODO: implement
        ...


class SentimentEvaluator:
    """
    Evaluates a SentimentContract using a Hugging Face sentiment pipeline.

    The pipeline is shared across all evaluations (lazy-loaded on first call)
    to avoid loading the model repeatedly.
    """

    _pipeline = None  # class-level shared pipeline instance

    async def evaluate(
        self,
        contract: SentimentContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        """
        Score the response content with a sentiment model and check against
        the minimum threshold.

        Args:
            contract: A SentimentContract with min_score and model name.
            request:  Normalised request body.
            response: Normalised response body.

        Returns:
            EvaluationResult(passed=False) if sentiment < min_score.
        """
        # TODO: implement — load pipeline if needed, run in thread pool, compare
        ...


class SchemaEvaluator:
    """
    Evaluates a SchemaContract using the jsonschema library.
    """

    async def evaluate(
        self,
        contract: SchemaContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        """
        Parse the response field specified by contract.target_field and
        validate it against contract.json_schema.

        Args:
            contract: A SchemaContract with json_schema and optional target_field.
            request:  Normalised request body.
            response: Normalised response body.

        Returns:
            EvaluationResult(passed=False) with jsonschema errors if invalid.
        """
        # TODO: implement — extract target, jsonschema.validate(), catch ValidationError
        ...


class CompositeEvaluator:
    """
    Evaluates a CompositeContract by recursively evaluating sub-contracts and
    applying the AND / OR operator.
    """

    async def evaluate(
        self,
        contract: CompositeContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        """
        Evaluate all sub-contracts and combine results with AND or OR logic.

        AND: Contract passes only when ALL sub-contracts pass.
        OR:  Contract fails when ANY sub-contract fails.

        Args:
            contract: A CompositeContract with contracts list and operator.
            request:  Normalised request body.
            response: Normalised response body.

        Returns:
            Aggregated EvaluationResult reflecting the combined outcome.
        """
        # TODO: implement — gather sub-results, apply operator
        ...


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

async def evaluate_contract(
    contract: AnyContract,
    request: dict,
    response: dict,
) -> EvaluationResult:
    """
    Route `contract` to the appropriate evaluator based on its type discriminator.

    This is the single entry point used by ContractRegistry.evaluate().

    Args:
        contract: Any concrete contract subtype.
        request:  Normalised request body.
        response: Normalised response body.

    Returns:
        EvaluationResult from the matched evaluator.

    Raises:
        ValueError: If the contract type is not recognised.
    """
    # TODO: implement dispatch table keyed by ContractType enum values
    ...
