"""
tests package

Unit and integration test suite for the AI Gateway.

Each test module targets a single gateway layer.  Tests are kept isolated from
external services by using pytest fixtures (conftest.py) that provide:
  - A pre-configured FastAPI TestClient.
  - In-memory Redis mock (fakeredis).
  - Dummy FAISS index and Embedder stubs.
  - Mocked provider adapters that return canned responses.
  - Pre-built contract definitions for contract evaluator tests.
"""
