"""
gateway/contracts package

Behavioral contract evaluation for LLM responses.

A "behavioral contract" is a declarative rule that an application registers
with the gateway to enforce invariants on every response it receives.
Examples:
  - "This assistant must never mention competitor products."
  - "All responses must include a disclaimer if they contain medical advice."
  - "The response sentiment score must be >= 0.3 (not hostile)."
  - "JSON responses must conform to this JSON Schema."

The package has three components:
  schema.py    — Pydantic models for defining contracts (the DSL).
  evaluator.py — Runs a contract against a (request, response) pair.
  registry.py  — Loads, stores, and dispatches contracts per application ID.

Public interface:
    from gateway.contracts.registry import ContractRegistry
    result = await registry.evaluate(app_id, request_body, response_body)
"""
