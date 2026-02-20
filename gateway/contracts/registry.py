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
            raise HTTPException(422, detail={"violations": report.violations})
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
        Files that fail to parse are logged and skipped (non-fatal).

        Replaces the current in-memory registry atomically.
        """
        # TODO: implement — glob for *.json and *.yaml, parse, populate self._registry
        ...

    def reload(self, app_id: str) -> bool:
        """
        Hot-reload the contract file for a specific app_id from disk.

        Args:
            app_id: The application identifier to reload.

        Returns:
            True if the file was found and reloaded successfully.
            False if no file exists for this app_id.
        """
        # TODO: implement
        ...

    async def evaluate(
        self,
        app_id: str,
        request: dict,
        response: dict,
    ) -> ContractReport:
        """
        Evaluate all enabled contracts for `app_id` against a request/response.

        If no contracts are registered for `app_id`, returns an unblocked
        ContractReport immediately.

        Steps:
          1. Look up the ContractDefinition for app_id.
          2. Iterate enabled contracts, calling evaluate_contract() on each.
          3. Collect violations; determine if any are BLOCK-action.
          4. Return a ContractReport.

        Args:
            app_id:   The application identifier (from API key or X-App-ID header).
            request:  Normalised request body dict.
            response: Normalised provider response body dict.

        Returns:
            A ContractReport with all violation details and the final blocked flag.
        """
        # TODO: implement — gather results via asyncio.gather()
        ...

    def register(self, definition: ContractDefinition) -> None:
        """
        Programmatically add or replace a ContractDefinition in the registry.

        Useful in tests and for dynamic contract management via an admin API.

        Args:
            definition: A valid ContractDefinition object.
        """
        # TODO: implement
        ...

    def get(self, app_id: str) -> Optional[ContractDefinition]:
        """
        Retrieve the ContractDefinition for a given app_id.

        Args:
            app_id: The application identifier.

        Returns:
            The ContractDefinition, or None if not registered.
        """
        # TODO: implement
        ...

    def list_apps(self) -> list[str]:
        """
        Return the list of all registered application IDs.

        Returns:
            Sorted list of app_id strings.
        """
        # TODO: implement
        ...

    def _parse_file(self, path: Path) -> Optional[ContractDefinition]:
        """
        Parse a single JSON contract file into a ContractDefinition.

        Args:
            path: Absolute path to the JSON file.

        Returns:
            A ContractDefinition on success, or None if parsing fails.
        """
        # TODO: implement — json.loads(), ContractDefinition.model_validate()
        ...
