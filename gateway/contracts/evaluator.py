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

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from gateway.contracts.schema import (
    AnyContract,
    CompositeContract,
    ContractAction,
    ContractType,
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
# Helper: extract response text
# ---------------------------------------------------------------------------

def _extract_response_text(response: dict) -> str:
    """
    Extract the assistant message content from a normalised response dict.

    Args:
        response: Normalised OpenAI-schema response dict.

    Returns:
        The assistant's reply as a plain string, or "" if not found.
    """
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _resolve_target_field(response: dict, target_field: Optional[str]) -> Any:
    """
    Navigate a dot-notation path (e.g. "choices.0.message.content") through
    a nested dict/list structure.

    Args:
        response:     The response dict to traverse.
        target_field: Dot-separated path, or None to return the whole response.

    Returns:
        The value at the specified path, or the entire response if path is None.

    Raises:
        KeyError / IndexError: If the path does not exist.
    """
    if not target_field:
        return response

    obj: Any = response
    for part in target_field.split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]  # numeric index into list
        else:
            obj = obj[part]
    return obj


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

        Matching behaviour:
          - case_sensitive=False (default): converts both sides to lowercase.
          - match_whole_word=True: wraps keyword in \\b word-boundary anchors.

        Args:
            contract: A KeywordContract with the keywords list and flags.
            request:  Normalised request body (unused by this evaluator).
            response: Normalised response body; checks choices[0].message.content.

        Returns:
            EvaluationResult(passed=False) if any keyword is found, else passed=True.
        """
        text = _extract_response_text(response)
        search_text = text if contract.case_sensitive else text.lower()

        for keyword in contract.keywords:
            needle = keyword if contract.case_sensitive else keyword.lower()

            if contract.match_whole_word:
                # Use a regex word-boundary search for whole-word matching.
                pattern = r"\b" + re.escape(needle) + r"\b"
                match = re.search(pattern, search_text)
                found = match is not None
            else:
                found = needle in search_text

            if found:
                return EvaluationResult(
                    passed=False,
                    action=contract.action,
                    contract_id=contract.id,
                    violation=ViolationDetail(
                        contract_id=contract.id,
                        message=f"Response contains forbidden keyword: {keyword!r}",
                        evidence=keyword,
                        location="choices[0].message.content",
                    ),
                )

        return EvaluationResult(passed=True, action=None, contract_id=contract.id)


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

        The contract is violated when the pattern MATCHES (re.search returns
        a result).  This means: "alert if the pattern appears in the response."

        Args:
            contract: A RegexContract with pattern and optional flags list.
            request:  Normalised request body.
            response: Normalised response body.

        Returns:
            EvaluationResult(passed=False) if the pattern matches, else passed=True.
        """
        text = _extract_response_text(response)

        # Build the combined flags integer from the list of flag names.
        combined_flags = 0
        for flag_name in contract.flags:
            flag = getattr(re, flag_name.upper(), None)
            if flag is not None:
                combined_flags |= flag

        compiled = re.compile(contract.pattern, combined_flags)
        match = compiled.search(text)

        if match:
            return EvaluationResult(
                passed=False,
                action=contract.action,
                contract_id=contract.id,
                violation=ViolationDetail(
                    contract_id=contract.id,
                    message=f"Response matches forbidden pattern: {contract.pattern!r}",
                    evidence=match.group(0)[:200],   # cap evidence at 200 chars
                    location=f"choices[0].message.content[{match.start()}:{match.end()}]",
                ),
            )

        return EvaluationResult(passed=True, action=None, contract_id=contract.id)


class SentimentEvaluator:
    """
    Evaluates a SentimentContract using a Hugging Face sentiment pipeline.

    The pipeline is shared across all evaluations (lazy-loaded on first call)
    to avoid loading the model repeatedly.

    The HuggingFace pipeline maps text to one of POSITIVE/NEUTRAL/NEGATIVE.
    We convert this to a numeric score: POSITIVE=+1, NEUTRAL=0, NEGATIVE=-1,
    which is then compared against contract.min_score.
    """

    _pipeline = None  # class-level shared pipeline instance
    _pipeline_model: str = ""  # track which model the pipeline was loaded for

    async def evaluate(
        self,
        contract: SentimentContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        """
        Score the response content with a sentiment model and check against
        the minimum threshold.

        The sentiment pipeline runs in a thread pool to avoid blocking the
        event loop during inference.

        Args:
            contract: A SentimentContract with min_score and model name.
            request:  Normalised request body.
            response: Normalised response body.

        Returns:
            EvaluationResult(passed=False) if sentiment < min_score.
        """
        text = _extract_response_text(response)
        if not text:
            return EvaluationResult(passed=True, action=None, contract_id=contract.id)

        loop = asyncio.get_event_loop()

        def _run_sentiment() -> float:
            # Lazy-load the pipeline (blocking, happens once per process).
            if (
                SentimentEvaluator._pipeline is None
                or SentimentEvaluator._pipeline_model != contract.model
            ):
                from transformers import pipeline as hf_pipeline  # type: ignore[import]
                SentimentEvaluator._pipeline = hf_pipeline(
                    "sentiment-analysis", model=contract.model
                )
                SentimentEvaluator._pipeline_model = contract.model

            result = SentimentEvaluator._pipeline(text[:512])[0]  # cap at 512 chars
            label = result.get("label", "").upper()
            score = result.get("score", 0.5)

            # Map label + confidence to a numeric compound score.
            # POSITIVE → +score, NEGATIVE → -score, NEUTRAL → 0
            if "POSITIVE" in label or "POS" in label:
                return score
            elif "NEGATIVE" in label or "NEG" in label:
                return -score
            else:
                return 0.0

        try:
            compound_score = await loop.run_in_executor(None, _run_sentiment)
        except Exception as exc:
            # If the sentiment model fails, pass the contract (fail open).
            return EvaluationResult(passed=True, action=None, contract_id=contract.id)

        if compound_score < contract.min_score:
            return EvaluationResult(
                passed=False,
                action=contract.action,
                contract_id=contract.id,
                violation=ViolationDetail(
                    contract_id=contract.id,
                    message=(
                        f"Response sentiment score {compound_score:.3f} is below "
                        f"minimum threshold {contract.min_score:.3f}"
                    ),
                    evidence=f"sentiment_score={compound_score:.3f}",
                    location="choices[0].message.content",
                ),
            )

        return EvaluationResult(passed=True, action=None, contract_id=contract.id)


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

        If target_field is set (e.g. "choices.0.message.content"), the value
        at that path is extracted and validated.  If None, the entire response
        dict is validated.

        Note: If the content is a JSON string (common with structured-output
        prompts), we try to parse it before validation.

        Args:
            contract: A SchemaContract with json_schema and optional target_field.
            request:  Normalised request body.
            response: Normalised response body.

        Returns:
            EvaluationResult(passed=False) with jsonschema errors if invalid.
        """
        import json as _json
        try:
            import jsonschema  # type: ignore[import]
        except ImportError:
            # jsonschema not installed — pass the contract (fail open).
            return EvaluationResult(passed=True, action=None, contract_id=contract.id)

        # Extract the target value (field or whole response).
        try:
            target = _resolve_target_field(response, contract.target_field)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return EvaluationResult(
                passed=False,
                action=contract.action,
                contract_id=contract.id,
                violation=ViolationDetail(
                    contract_id=contract.id,
                    message=f"Could not extract target field {contract.target_field!r}: {exc}",
                    location=contract.target_field,
                ),
            )

        # If the target is a string, try to parse it as JSON (structured output).
        if isinstance(target, str):
            try:
                target = _json.loads(target)
            except _json.JSONDecodeError:
                pass  # Validate the raw string against the schema

        # Validate against the JSON Schema.
        try:
            jsonschema.validate(instance=target, schema=contract.json_schema)
        except jsonschema.ValidationError as exc:
            return EvaluationResult(
                passed=False,
                action=contract.action,
                contract_id=contract.id,
                violation=ViolationDetail(
                    contract_id=contract.id,
                    message=f"JSON Schema validation failed: {exc.message}",
                    evidence=str(exc.instance)[:200],
                    location=contract.target_field or "response",
                ),
            )
        except jsonschema.SchemaError as exc:
            # The schema itself is invalid — pass the contract and log.
            return EvaluationResult(passed=True, action=None, contract_id=contract.id)

        return EvaluationResult(passed=True, action=None, contract_id=contract.id)


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

        OR  logic (default): The composite FAILS when ANY sub-contract fails.
                             This is the most common use — "block if any rule fires."
        AND logic:           The composite FAILS only when ALL sub-contracts fail.
                             Useful for "block only when multiple conditions coexist."

        Sub-contracts are evaluated concurrently via asyncio.gather() for
        efficiency when multiple slow (sentiment, schema) checks are combined.

        Args:
            contract: A CompositeContract with contracts list and operator.
            request:  Normalised request body.
            response: Normalised response body.

        Returns:
            Aggregated EvaluationResult reflecting the combined outcome.
        """
        # Evaluate all sub-contracts concurrently.
        sub_results: list[EvaluationResult] = await asyncio.gather(
            *[evaluate_contract(sub, request, response) for sub in contract.contracts]
        )

        failures = [r for r in sub_results if not r.passed]

        if contract.operator == "OR":
            # OR: fail if ANY sub-contract fails.
            if failures:
                return EvaluationResult(
                    passed=False,
                    action=contract.action,
                    contract_id=contract.id,
                    violation=ViolationDetail(
                        contract_id=contract.id,
                        message=(
                            f"Composite (OR) contract violated: "
                            f"{len(failures)}/{len(sub_results)} sub-contracts failed"
                        ),
                        evidence=failures[0].violation.message if failures[0].violation else None,
                    ),
                )
        else:
            # AND: fail only if ALL sub-contracts fail.
            if len(failures) == len(sub_results):
                return EvaluationResult(
                    passed=False,
                    action=contract.action,
                    contract_id=contract.id,
                    violation=ViolationDetail(
                        contract_id=contract.id,
                        message=(
                            f"Composite (AND) contract violated: "
                            f"all {len(sub_results)} sub-contracts failed"
                        ),
                    ),
                )

        return EvaluationResult(passed=True, action=None, contract_id=contract.id)


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

# Dispatch table: contract type → evaluator instance.
# Instances are created once (they're stateless) and reused.
_EVALUATORS: dict[str, Any] = {
    ContractType.KEYWORD:   KeywordEvaluator(),
    ContractType.REGEX:     RegexEvaluator(),
    ContractType.SENTIMENT: SentimentEvaluator(),
    ContractType.SCHEMA:    SchemaEvaluator(),
    ContractType.COMPOSITE: CompositeEvaluator(),
}


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
    evaluator = _EVALUATORS.get(contract.type)
    if evaluator is None:
        raise ValueError(f"No evaluator registered for contract type: {contract.type!r}")

    return await evaluator.evaluate(contract, request, response)
