"""
gateway/contracts package

Behavioral contract evaluation for LLM responses — three-tier architecture.

Tier 1 — Deterministic (< 1ms):
  Length limits, keyword matching, regex, language detection, JSON schema.
  Always accurate, zero ambiguity.

Tier 2 — Classifier (~10-15ms):
  Sentiment analysis, topic boundary enforcement via zero-shot classification.
  Uses lightweight transformer models on CPU.

Tier 3 — LLM Judge (~100-300ms):
  Complex natural-language assertions evaluated by a small LLM.
  Runs asynchronously as FLAG contracts — never blocks the response.

Drift Detection:
  Every evaluation produces a compliance score (0.0-1.0) that is stored in
  a Redis time series.  When rolling compliance drops >10% relative to the
  7-day baseline, a DriftAlert is generated.

Public interface:
    from gateway.contracts.registry import ContractRegistry
    from gateway.contracts.drift import DriftDetector

    report = await registry.evaluate(app_id, request_body, response_body)
    alerts = await drift_detector.check_drift(app_id, contract_id)
"""

from gateway.contracts.schema import (
    ContractType,
    ContractAction,
    EvaluationTier,
    ContractDefinition,
    AnyContract,
)
from gateway.contracts.evaluator import EvaluationResult, ViolationDetail
from gateway.contracts.registry import ContractRegistry, ContractReport
from gateway.contracts.drift import DriftDetector, DriftAlert, ComplianceSnapshot

__all__ = [
    "ContractType",
    "ContractAction",
    "EvaluationTier",
    "ContractDefinition",
    "AnyContract",
    "EvaluationResult",
    "ViolationDetail",
    "ContractRegistry",
    "ContractReport",
    "DriftDetector",
    "DriftAlert",
    "ComplianceSnapshot",
]
