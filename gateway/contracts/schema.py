"""
gateway/contracts/schema.py

Pydantic models that form the behavioral contract definition language (DSL).

Contract types — three evaluation tiers:

  Deterministic (< 1ms, always correct):
    - KeywordContract     — block responses containing forbidden keywords
    - RegexContract       — match response text against a compiled regex
    - LengthLimitContract — enforce word/character/sentence limits
    - LanguageMatchContract — verify response language matches request language

  Classifier (~10-15ms, semantic understanding):
    - SentimentContract      — require response sentiment above a threshold
    - TopicBoundaryContract  — verify response stays within allowed topics

  LLM Judge (~100-300ms, nuanced reasoning):
    - LLMJudgeContract    — evaluate complex assertions via a small LLM

  Structural:
    - SchemaContract      — validate response JSON against a JSON Schema
    - CompositeContract   — combine multiple contracts with AND / OR logic
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EvaluationTier(str, Enum):
    """
    Classification of contract evaluation cost.  Used to decide execution
    strategy: deterministic checks run synchronously for BLOCK actions,
    expensive tiers run in background for FLAG actions.
    """
    DETERMINISTIC = "deterministic"
    CLASSIFIER = "classifier"
    LLM_JUDGE = "llm_judge"


class ContractType(str, Enum):
    KEYWORD = "keyword"
    REGEX = "regex"
    SENTIMENT = "sentiment"
    SCHEMA = "schema"
    COMPOSITE = "composite"
    LENGTH_LIMIT = "length_limit"
    LANGUAGE_MATCH = "language_match"
    TOPIC_BOUNDARY = "topic_boundary"
    LLM_JUDGE = "llm_judge"


class ContractAction(str, Enum):
    """
    Action to take when a contract is violated.

    BLOCK: Reject the response synchronously — the upstream response is NOT
           returned to the caller.  Only use with DETERMINISTIC or CLASSIFIER
           tier contracts to keep latency low.
    FLAG:  Log the violation and fire an alert, but return the response to
           the caller immediately.  Expensive evaluations (LLM_JUDGE) should
           always use FLAG so the user never waits for them.
    LOG:   Record the violation silently; caller is unaware.  Useful for
           monitoring compliance without any user-visible effect.
    """
    BLOCK = "block"
    FLAG = "flag"
    LOG = "log"


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class BaseContract(BaseModel):
    """
    Common fields shared by all contract types.

    Every contract has an evaluation_tier that determines its execution
    strategy.  Subclasses set this as a class-level default — callers
    should not override it unless they know what they're doing.
    """
    id: str
    name: str
    action: ContractAction = ContractAction.BLOCK
    enabled: bool = True
    description: Optional[str] = None
    evaluation_tier: EvaluationTier = EvaluationTier.DETERMINISTIC


# ---------------------------------------------------------------------------
# Deterministic contracts (< 1ms)
# ---------------------------------------------------------------------------

class KeywordContract(BaseContract):
    """Block or flag when the response contains any listed keyword."""
    type: Literal[ContractType.KEYWORD] = ContractType.KEYWORD
    evaluation_tier: EvaluationTier = EvaluationTier.DETERMINISTIC
    keywords: list[str] = Field(..., min_length=1)
    case_sensitive: bool = False
    match_whole_word: bool = False


class RegexContract(BaseContract):
    """Evaluate response text against a regular expression pattern."""
    type: Literal[ContractType.REGEX] = ContractType.REGEX
    evaluation_tier: EvaluationTier = EvaluationTier.DETERMINISTIC
    pattern: str
    flags: list[str] = Field(default_factory=list)

    @field_validator("pattern")
    @classmethod
    def validate_regex(cls, v: str) -> str:
        import re
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern {v!r}: {exc}") from exc
        return v


class LengthLimitContract(BaseContract):
    """
    Enforce length constraints on the response content.

    Supports word count, character count, and sentence count limits.
    All three can be set simultaneously — all specified limits must pass.
    """
    type: Literal[ContractType.LENGTH_LIMIT] = ContractType.LENGTH_LIMIT
    evaluation_tier: EvaluationTier = EvaluationTier.DETERMINISTIC
    max_words: Optional[int] = Field(None, ge=1)
    max_characters: Optional[int] = Field(None, ge=1)
    max_sentences: Optional[int] = Field(None, ge=1)

    @field_validator("max_words", "max_characters", "max_sentences")
    @classmethod
    def at_least_one_limit(cls, v, info):
        return v


class LanguageMatchContract(BaseContract):
    """
    Verify that the response language matches the user's message language.

    Uses langdetect for language identification.  Optionally accepts an
    explicit expected_language ISO code to enforce instead of auto-detection.
    """
    type: Literal[ContractType.LANGUAGE_MATCH] = ContractType.LANGUAGE_MATCH
    evaluation_tier: EvaluationTier = EvaluationTier.DETERMINISTIC
    expected_language: Optional[str] = Field(
        None,
        description="ISO 639-1 code (e.g. 'en', 'fr'). If None, auto-detects from user message.",
    )
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Classifier contracts (~10-15ms)
# ---------------------------------------------------------------------------

class SentimentContract(BaseContract):
    """Require response sentiment to meet a minimum positivity threshold."""
    type: Literal[ContractType.SENTIMENT] = ContractType.SENTIMENT
    evaluation_tier: EvaluationTier = EvaluationTier.CLASSIFIER
    min_score: float = Field(0.0, ge=-1.0, le=1.0)
    model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class TopicBoundaryContract(BaseContract):
    """
    Verify that the response stays within a set of allowed topics.

    Uses a zero-shot classification pipeline (NLI-based) so no custom
    training is required — you just list the allowed topic labels.
    The classifier scores the response against each label and checks
    that the top label is in the allowed set with confidence >= threshold.
    """
    type: Literal[ContractType.TOPIC_BOUNDARY] = ContractType.TOPIC_BOUNDARY
    evaluation_tier: EvaluationTier = EvaluationTier.CLASSIFIER
    allowed_topics: list[str] = Field(..., min_length=1)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    model: str = "facebook/bart-large-mnli"
    off_topic_label: str = Field(
        "other",
        description="Label appended to candidate list to catch off-topic responses",
    )


# ---------------------------------------------------------------------------
# LLM Judge contracts (~100-300ms)
# ---------------------------------------------------------------------------

class LLMJudgeContract(BaseContract):
    """
    Evaluate a natural-language assertion about the response using a small LLM.

    The assertion is a plain English statement like:
      "Response does not provide specific investment or financial advice"

    The judge model receives the assertion + response and returns a compliance
    score from 0.0 (clear violation) to 1.0 (fully compliant).

    IMPORTANT: LLM judge contracts should use action=FLAG (not BLOCK) because
    they add 100-300ms of latency.  The architecture runs FLAG contracts
    asynchronously so the user never waits.
    """
    type: Literal[ContractType.LLM_JUDGE] = ContractType.LLM_JUDGE
    evaluation_tier: EvaluationTier = EvaluationTier.LLM_JUDGE
    assertion: str = Field(..., min_length=5)
    threshold: float = Field(0.7, ge=0.0, le=1.0)
    judge_model: str = Field(
        "gpt-4o-mini",
        description="Model alias to use as the judge (routed through the gateway's own provider system)",
    )
    system_prompt: str = Field(
        default=(
            "You are a compliance evaluator. You will be given an assertion about "
            "expected behavior and a response to evaluate. Score how well the response "
            "complies with the assertion. Respond with ONLY a JSON object: "
            '{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}'
        ),
    )


# ---------------------------------------------------------------------------
# Structural contracts
# ---------------------------------------------------------------------------

class SchemaContract(BaseContract):
    """Validate response body (or a field within it) against a JSON Schema."""
    type: Literal[ContractType.SCHEMA] = ContractType.SCHEMA
    evaluation_tier: EvaluationTier = EvaluationTier.DETERMINISTIC
    json_schema: dict[str, Any]
    target_field: Optional[str] = None


class CompositeContract(BaseContract):
    """Combine multiple contracts with boolean AND or OR logic."""
    type: Literal[ContractType.COMPOSITE] = ContractType.COMPOSITE
    operator: Literal["AND", "OR"] = "OR"
    contracts: list["AnyContract"]


# Discriminated union for all concrete contract types.
AnyContract = Union[
    KeywordContract,
    RegexContract,
    LengthLimitContract,
    LanguageMatchContract,
    SentimentContract,
    TopicBoundaryContract,
    LLMJudgeContract,
    SchemaContract,
    CompositeContract,
]

CompositeContract.model_rebuild()


# ---------------------------------------------------------------------------
# Top-level definition
# ---------------------------------------------------------------------------

class ContractDefinition(BaseModel):
    """
    A complete contract specification for a single application.

    Loaded from a JSON file by ContractRegistry and associated with
    an application ID.
    """
    app_id: str
    version: str = "1.0.0"
    contracts: list[AnyContract] = Field(default_factory=list)
