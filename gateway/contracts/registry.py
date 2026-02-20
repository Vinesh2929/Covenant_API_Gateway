"""
gateway/contracts/registry.py

Loads, stores, and dispatches behavioral contracts per application.

Responsibilities:
  - Scan a contracts directory on startup and load all ContractDefinition
    files (JSON or YAML) into an in-memory registry keyed by app_id.
  - Expose an evaluate() method that runs every enabled contract for a given
    app_id against a (request, response) pair and aggregates the results.
  - Support hot-reload: a file watcher (or manual trigger via admin endpoint)
    can reload a single app's contracts without restarting the gateway.
  - Handle app_ids with no registered contracts gracefully (pass-through).
  - Return an aggregated ContractReport that lists every violation so it can
    be included in Langfuse trace metadata.

Key classes / functions:
  - ContractReport           — dataclass: app_id, blocked, violations list, evaluation_ms
  - ContractRegistry         — main class
    - __init__(contracts_dir) — store directory path, do NOT load yet
    - load()                  — scan directory and parse all contract files
    - reload(app_id)          — hot-reload contracts for a specific app
    - evaluate(app_id, req, resp) — async: run all contracts, return ContractReport
    - register(definition)    — programmatically register a ContractDefinition
    - get(app_id)             — return ContractDefinition for app_id or None
    - list_apps()             — return all registered app_ids
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from gateway.contracts.schema import ContractDefinition, ContractAction
from gateway.contracts.evaluator import EvaluationResult, evaluate_contract


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContractReport:
    """
    Aggregated result of evaluating all contracts for one request-response pair.

    Attributes:
        app_id:         The application whose contracts were evaluated.
        blocked:        True if at least one BLOCK-action contract was violated.
        violations:     List of EvaluationResult objects for violated contracts.
        evaluation_ms:  Total wall-clock time for all contract evaluations.
    """
    app_id: str
    blocked: bool
    violations: list[EvaluationResult] = field(default_factory=list)
    evaluation_ms: float = 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ContractRegistry:
    """
    Central store for all application behavioral contract definitions.

    Contract files are expected to be JSON files in the contracts directory
    with filenames matching their app_id (e.g. `my-app.json`).

    The registry is thread-safe for reads; writes (load/reload/register) use
    a simple dict swap to avoid locking.

    Usage::

        registry = ContractRegistry("contracts/")
        registry.load()

        report = await registry.evaluate(app_id, request_body, response_body)
        if report.blocked:
            raise HTTPException(422, detail={"violations": [...]})
    """

    def __init__(self, contracts_dir: str) -> None:
        """
        Store the contracts directory path.  Does NOT scan the filesystem yet.

        Args:
            contracts_dir: Path to the directory containing JSON contract files.
        """
        self._contracts_dir = Path(contracts_dir)
        self._registry: dict[str, ContractDefinition] = {}

    def load(self) -> None:
        """
        Scan the contracts directory and parse every *.json file found.

        Each file should contain a valid ContractDefinition JSON object.
        Files that fail to parse are logged and skipped (non-fatal), so a
        single bad contract file cannot bring down the gateway.

        The resulting registry is swapped in atomically (whole-dict replacement)
        so in-flight requests always see a consistent snapshot.
        """
        if not self._contracts_dir.exists():
            # Contracts directory doesn't exist yet — start with empty registry.
            return

        new_registry: dict[str, ContractDefinition] = {}

        # Glob for JSON contract files (YAML support can be added here later).
        for path in sorted(self._contracts_dir.glob("*.json")):
            definition = self._parse_file(path)
            if definition is not None:
                new_registry[definition.app_id] = definition

        # Atomic swap: replace the entire registry in one assignment.
        # CPython's GIL makes dict assignment atomic for reading threads.
        self._registry = new_registry

    def reload(self, app_id: str) -> bool:
        """
        Hot-reload the contract file for a specific app_id from disk.

        Looks for a file named <app_id>.json in the contracts directory.
        Replaces only the single app's entry in the registry.

        Args:
            app_id: The application identifier to reload.

        Returns:
            True if the file was found and reloaded successfully.
            False if no file exists for this app_id.
        """
        # The expected filename is <app_id>.json in the contracts directory.
        path = self._contracts_dir / f"{app_id}.json"

        if not path.exists():
            return False

        definition = self._parse_file(path)
        if definition is None:
            return False

        # Partial update — only replace the single app's entry.
        self._registry[definition.app_id] = definition
        return True

    async def evaluate(
        self,
        app_id: str,
        request: dict,
        response: dict,
    ) -> ContractReport:
        """
        Evaluate all enabled contracts for `app_id` against a request/response.

        If no contracts are registered for `app_id`, returns an unblocked
        ContractReport immediately (pass-through).

        Steps:
          1. Look up the ContractDefinition for app_id.
          2. Filter to only enabled contracts.
          3. Evaluate all enabled contracts concurrently via asyncio.gather().
          4. Collect violations; determine if any have action=BLOCK.
          5. Return a ContractReport.

        Args:
            app_id:   The application identifier (from API key or X-App-ID header).
            request:  Normalised request body dict.
            response: Normalised provider response body dict.

        Returns:
            A ContractReport with all violation details and the final blocked flag.
        """
        start = time.perf_counter()

        definition = self._registry.get(app_id)
        if definition is None:
            # No contracts registered for this app — pass through immediately.
            return ContractReport(app_id=app_id, blocked=False, evaluation_ms=0.0)

        # Filter to only enabled contracts before evaluation.
        enabled_contracts = [c for c in definition.contracts if c.enabled]

        if not enabled_contracts:
            return ContractReport(app_id=app_id, blocked=False, evaluation_ms=0.0)

        # Evaluate all contracts concurrently for efficiency.
        # Contracts that use I/O (sentiment model, schema validation) benefit
        # most from concurrency — a composite contract with 5 rules runs in
        # the time of the slowest rule, not the sum.
        results: list[EvaluationResult] = await asyncio.gather(
            *[evaluate_contract(contract, request, response) for contract in enabled_contracts]
        )

        # Separate violations from passes.
        violations = [r for r in results if not r.passed]

        # The request is blocked if ANY violated contract has action=BLOCK.
        # WARN and LOG violations are collected but don't block the response.
        blocked = any(
            v.action == ContractAction.BLOCK
            for v in violations
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return ContractReport(
            app_id=app_id,
            blocked=blocked,
            violations=violations,
            evaluation_ms=elapsed_ms,
        )

    def register(self, definition: ContractDefinition) -> None:
        """
        Programmatically add or replace a ContractDefinition in the registry.

        Useful in tests and for dynamic contract management via an admin API.

        Args:
            definition: A valid ContractDefinition object.
        """
        self._registry[definition.app_id] = definition

    def get(self, app_id: str) -> Optional[ContractDefinition]:
        """
        Retrieve the ContractDefinition for a given app_id.

        Args:
            app_id: The application identifier.

        Returns:
            The ContractDefinition, or None if not registered.
        """
        return self._registry.get(app_id)

    def list_apps(self) -> list[str]:
        """
        Return the list of all registered application IDs.

        Returns:
            Sorted list of app_id strings.
        """
        return sorted(self._registry.keys())

    def _parse_file(self, path: Path) -> Optional[ContractDefinition]:
        """
        Parse a single JSON contract file into a ContractDefinition.

        Uses Pydantic's model_validate() which handles type coercion and
        validates the discriminated union (AnyContract) automatically.

        Args:
            path: Absolute path to the JSON file.

        Returns:
            A ContractDefinition on success, or None if parsing fails.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            # model_validate() is the Pydantic v2 way to instantiate from a dict.
            # It handles the AnyContract discriminated union automatically using
            # the "type" field as the discriminator.
            return ContractDefinition.model_validate(raw)

        except json.JSONDecodeError as exc:
            # Malformed JSON — skip this file.
            return None
        except Exception as exc:
            # Pydantic validation error, unexpected field, etc. — skip.
            return None
