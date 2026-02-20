"""
gateway/contracts/registry.py

Loads, stores, and dispatches behavioral contracts per application.

The key architectural change in v2: tiered execution.

  BLOCK contracts run synchronously — the response is NOT returned until
  all BLOCK evaluations complete.  These should only be deterministic or
  classifier tier checks to keep latency under ~15ms.

  FLAG/LOG contracts run asynchronously — the response is returned to the
  caller immediately and the evaluations run in a background task.  Results
  are logged to Langfuse and fed into the drift detector.

This separation means:
  - Users experience at most ~15ms of added latency (deterministic + classifier)
  - Expensive LLM judge evaluations never block the response
  - Drift detection data is collected on every request without cost
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

from gateway.contracts.schema import (
    AnyContract,
    ContractAction,
    ContractDefinition,
    EvaluationTier,
)
from gateway.contracts.evaluator import EvaluationResult, evaluate_contract
from gateway.contracts.drift import DriftDetector

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContractReport:
    """
    Aggregated result of evaluating all contracts for one request-response pair.

    In v2, this reports the results of BLOCK contracts only (the synchronous
    path).  FLAG/LOG results arrive later via the background task and are
    available through the drift detection API and Langfuse traces.
    """
    app_id: str
    blocked: bool
    violations: list[EvaluationResult] = field(default_factory=list)
    warnings: list[EvaluationResult] = field(default_factory=list)
    evaluation_ms: float = 0.0
    block_count: int = 0
    flag_count: int = 0
    total_contracts: int = 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ContractRegistry:
    """
    Central store and execution engine for behavioral contracts.

    The evaluate() method implements the tiered execution model:
      1. Split contracts by action: BLOCK (sync) vs FLAG/LOG (async).
      2. Run all BLOCK contracts in parallel — wait for completion.
      3. If any BLOCK contract fails → return blocked report immediately.
      4. Fire FLAG/LOG contracts as a background asyncio.Task — don't wait.
      5. Background task records compliance scores to the drift detector.
    """

    def __init__(
        self,
        contracts_dir: str,
        drift_detector: Optional[DriftDetector] = None,
    ) -> None:
        self._contracts_dir = Path(contracts_dir)
        self._registry: dict[str, ContractDefinition] = {}
        self._drift_detector = drift_detector
        self._background_tasks: set[asyncio.Task] = set()

    def load(self) -> None:
        """Scan the contracts directory and parse every *.json file."""
        if not self._contracts_dir.exists():
            return

        new_registry: dict[str, ContractDefinition] = {}

        for path in sorted(self._contracts_dir.glob("*.json")):
            definition = self._parse_file(path)
            if definition is not None:
                new_registry[definition.app_id] = definition

        self._registry = new_registry

    def reload(self, app_id: str) -> bool:
        """Hot-reload the contract file for a specific app_id."""
        path = self._contracts_dir / f"{app_id}.json"
        if not path.exists():
            return False

        definition = self._parse_file(path)
        if definition is None:
            return False

        self._registry[definition.app_id] = definition
        return True

    async def evaluate(
        self,
        app_id: str,
        request: dict,
        response: dict,
    ) -> ContractReport:
        """
        Evaluate all contracts with tiered execution.

        Execution model:
          1. BLOCK contracts: run synchronously in parallel, must all pass
             before the response is returned to the caller.
          2. FLAG/LOG contracts: fired as a background task after the
             response is returned.  Never blocks the caller.
        """
        start = time.perf_counter()

        definition = self._registry.get(app_id)
        if definition is None:
            return ContractReport(app_id=app_id, blocked=False, evaluation_ms=0.0)

        enabled = [c for c in definition.contracts if c.enabled]
        if not enabled:
            return ContractReport(app_id=app_id, blocked=False, evaluation_ms=0.0)

        # Split by action: BLOCK is synchronous, FLAG/LOG is async background
        block_contracts = [c for c in enabled if c.action == ContractAction.BLOCK]
        async_contracts = [c for c in enabled if c.action != ContractAction.BLOCK]

        # --- Synchronous path: evaluate all BLOCK contracts in parallel ---
        block_results: list[EvaluationResult] = []
        if block_contracts:
            block_results = await asyncio.gather(
                *[evaluate_contract(c, request, response) for c in block_contracts]
            )

        block_violations = [r for r in block_results if not r.passed]
        blocked = len(block_violations) > 0

        # Record compliance scores for BLOCK contracts (drift detection)
        if self._drift_detector:
            for result in block_results:
                try:
                    await self._drift_detector.record(
                        app_id, result.contract_id, result.compliance_score
                    )
                except Exception as exc:
                    log.debug("Drift record failed", error=str(exc))

        # --- Asynchronous path: fire FLAG/LOG contracts in background ---
        if async_contracts and not blocked:
            task = asyncio.create_task(
                self._evaluate_background(app_id, async_contracts, request, response)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Collect warnings from BLOCK results that passed but had low compliance
        warnings = [
            r for r in block_results
            if r.passed and r.compliance_score < 0.8
        ]

        return ContractReport(
            app_id=app_id,
            blocked=blocked,
            violations=block_violations,
            warnings=warnings,
            evaluation_ms=elapsed_ms,
            block_count=len(block_contracts),
            flag_count=len(async_contracts),
            total_contracts=len(enabled),
        )

    async def _evaluate_background(
        self,
        app_id: str,
        contracts: list[AnyContract],
        request: dict,
        response: dict,
    ) -> list[EvaluationResult]:
        """
        Run FLAG/LOG contracts in the background.

        Results are logged and recorded for drift detection.  The caller
        (the user's HTTP request) has already received the response.
        """
        results: list[EvaluationResult] = []

        try:
            results = await asyncio.gather(
                *[evaluate_contract(c, request, response) for c in contracts],
                return_exceptions=False,
            )
        except Exception as exc:
            log.error("Background contract evaluation failed", error=str(exc))
            return results

        # Record all compliance scores for drift detection
        for result in results:
            if self._drift_detector:
                try:
                    await self._drift_detector.record(
                        app_id, result.contract_id, result.compliance_score
                    )
                except Exception as exc:
                    log.debug("Drift record failed", error=str(exc))

            # Log violations for FLAG/LOG contracts
            if not result.passed:
                log.warning(
                    "Background contract violation",
                    app_id=app_id,
                    contract_id=result.contract_id,
                    action=result.action.value if result.action else "unknown",
                    tier=result.evaluation_tier.value,
                    compliance_score=result.compliance_score,
                    violation=result.violation.message if result.violation else None,
                    latency_ms=round(result.latency_ms, 2),
                )

        return results

    def register(self, definition: ContractDefinition) -> None:
        """Programmatically add or replace a ContractDefinition."""
        self._registry[definition.app_id] = definition

    def get(self, app_id: str) -> Optional[ContractDefinition]:
        return self._registry.get(app_id)

    def list_apps(self) -> list[str]:
        return sorted(self._registry.keys())

    def get_contract_ids(self, app_id: str) -> list[str]:
        """Return all contract IDs for an app (used by drift detector)."""
        definition = self._registry.get(app_id)
        if definition is None:
            return []
        return [c.id for c in definition.contracts if c.enabled]

    async def shutdown(self) -> None:
        """Wait for all background tasks to complete before shutdown."""
        if self._background_tasks:
            log.info(
                "Waiting for background contract tasks",
                pending=len(self._background_tasks),
            )
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

    def _parse_file(self, path: Path) -> Optional[ContractDefinition]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return ContractDefinition.model_validate(raw)
        except json.JSONDecodeError:
            log.warning("Malformed contract JSON", path=str(path))
            return None
        except Exception as exc:
            log.warning("Contract parse error", path=str(path), error=str(exc))
            return None
