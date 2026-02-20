"""
gateway/contracts/schema.py

Pydantic models that form the behavioral contract definition language (DSL).

Responsibilities:
  - Define the data models that application developers use to declare what
    invariants the gateway should enforce on their behalf.
  - Support multiple contract types: keyword blacklists, regex rules, JSON
    Schema validation, sentiment thresholds, and custom Python callables
    (advanced, loaded from a trusted plugin directory).
  - Provide sensible defaults and validation so that a minimal contract
    definition is easy to write while still supporting full expressiveness.
  - Be serialisable to/from JSON so contracts can be stored in files or a
    database and hot-reloaded without restarting the gateway.

Contract types implemented:
  - KeywordContract    — block responses containing any of a list of keywords
  - RegexContract      — match response text against a compiled regex
  - SentimentContract  — require response sentiment to be above a threshold
  - SchemaContract     — validate response JSON against a JSON Schema dict
  - CompositeContract  — combine multiple contracts with AND / OR logic

Key classes:
  - ContractType       — Enum of supported contract type names
  - ContractAction     — Enum: BLOCK, WARN, LOG (what to do when violated)
  - BaseContract       — abstract Pydantic base with id, name, action, enabled
  - KeywordContract    — keywords list + case_sensitive flag
  - RegexContract      — pattern string + description
  - SentimentContract  — min_score float, sentiment library selection
  - SchemaContract     — json_schema dict for jsonschema validation
  - CompositeContract  — contracts list + operator (AND/OR)
  - ContractDefinition — top-level wrapper: app_id + list of contracts
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContractType(str, Enum):
    """
    Discriminator values used to identify contract subtypes in a
    ContractDefinition list.
    """
    KEYWORD = "keyword"
    REGEX = "regex"
    SENTIMENT = "sentiment"
    SCHEMA = "schema"
    COMPOSITE = "composite"


class ContractAction(str, Enum):
    """
    Action to take when a contract is violated.

    BLOCK: Reject the response with HTTP 422 and surface the violation to
           the caller.  The upstream response is NOT returned.
    WARN:  Return the response but include an X-Contract-Warning header and
           log the violation.
    LOG:   Record the violation silently; caller is unaware.
    """
    BLOCK = "block"
    WARN = "warn"
    LOG = "log"


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class BaseContract(BaseModel):
    """
    Common fields shared by all contract types.

    Attributes:
        id:          Unique slug identifier for this contract rule.
        name:        Human-readable display name.
        action:      What to do when this contract is violated.
        enabled:     Set to False to disable without deleting the rule.
        description: Optional notes about why this contract exists.
    """
    id: str
    name: str
    action: ContractAction = ContractAction.BLOCK
    enabled: bool = True
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Concrete contract types
# ---------------------------------------------------------------------------

class KeywordContract(BaseContract):
    """
    Block or warn when the response text contains any of the listed keywords.

    Attributes:
        type:            Discriminator literal, always "keyword".
        keywords:        List of strings to search for.
        case_sensitive:  If False (default), matching is case-insensitive.
        match_whole_word: If True, only match whole words (not substrings).
    """
    type: Literal[ContractType.KEYWORD] = ContractType.KEYWORD
    keywords: list[str] = Field(..., min_length=1)
    case_sensitive: bool = False
    match_whole_word: bool = False


class RegexContract(BaseContract):
    """
    Evaluate the response text against a regular expression pattern.

    Attributes:
        type:     Discriminator literal, always "regex".
        pattern:  A Python regex string.  The contract is violated when the
                  pattern matches (i.e. re.search returns a result).
        flags:    Optional list of re flag names to apply (e.g. ["IGNORECASE"]).
    """
    type: Literal[ContractType.REGEX] = ContractType.REGEX
    pattern: str
    flags: list[str] = Field(default_factory=list)

    @field_validator("pattern")
    @classmethod
    def validate_regex(cls, v: str) -> str:
        """Ensure the pattern compiles without error."""
        # TODO: implement — import re, try re.compile(v), re-raise ValueError
        return v


class SentimentContract(BaseContract):
    """
    Require the response sentiment to meet a minimum positivity threshold.

    Uses a lightweight transformer-based sentiment classifier.  Sentiment
    scores range from -1.0 (very negative) to 1.0 (very positive).

    Attributes:
        type:       Discriminator literal, always "sentiment".
        min_score:  Minimum acceptable compound sentiment score (-1.0 to 1.0).
        model:      Sentiment model identifier (default: a lightweight HF model).
    """
    type: Literal[ContractType.SENTIMENT] = ContractType.SENTIMENT
    min_score: float = Field(0.0, ge=-1.0, le=1.0)
    model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class SchemaContract(BaseContract):
    """
    Validate the response body (or a named field within it) against a JSON
    Schema definition.

    Useful for structured-output prompts where the response must conform to a
    strict format.

    Attributes:
        type:        Discriminator literal, always "schema".
        json_schema: A JSON Schema dict (Draft 7 or later).
        target_field: Dot-notation path to the response field to validate
                      (e.g. "choices.0.message.content").  If None, validates
                      the entire response object.
    """
    type: Literal[ContractType.SCHEMA] = ContractType.SCHEMA
    json_schema: dict[str, Any]
    target_field: Optional[str] = None


class CompositeContract(BaseContract):
    """
    Combine multiple contracts with boolean AND or OR logic.

    AND: The composite is violated only when ALL sub-contracts are violated.
    OR:  The composite is violated when ANY sub-contract is violated.

    Attributes:
        type:      Discriminator literal, always "composite".
        operator:  "AND" or "OR".
        contracts: List of concrete sub-contract definitions.
    """
    type: Literal[ContractType.COMPOSITE] = ContractType.COMPOSITE
    operator: Literal["AND", "OR"] = "OR"
    contracts: list["AnyContract"]


# Discriminated union for all concrete contract types.
AnyContract = Union[
    KeywordContract,
    RegexContract,
    SentimentContract,
    SchemaContract,
    CompositeContract,
]

# Required for forward reference resolution (CompositeContract.contracts)
CompositeContract.model_rebuild()


# ---------------------------------------------------------------------------
# Top-level definition
# ---------------------------------------------------------------------------

class ContractDefinition(BaseModel):
    """
    A complete contract specification for a single application.

    Loaded from a YAML or JSON file by ContractRegistry and associated with
    an application ID (typically the API key prefix or an explicit app_id
    header).

    Attributes:
        app_id:    Identifier for the application these contracts apply to.
        version:   Semver string for the contract file.
        contracts: Ordered list of contract rules to evaluate on every response.
    """
    app_id: str
    version: str = "1.0.0"
    contracts: list[AnyContract] = Field(default_factory=list)
