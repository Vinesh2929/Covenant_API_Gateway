"""
gateway/providers package

Adapter layer that normalises LLM provider request/response formats to and
from a single internal OpenAI-compatible schema.

Each adapter handles:
  - Translating the internal normalised request dict into the provider's
    native API format (headers, body, URL).
  - Forwarding the request to the provider's HTTP endpoint.
  - Translating the provider's response back into the normalised format.
  - Extracting token usage from the response for cost tracking.
  - Handling provider-specific error codes and retrying transient errors.

Adapters:
  base.py              — abstract BaseProviderAdapter (Protocol + ABC)
  openai_adapter.py    — OpenAI and OpenAI-compatible endpoints (Azure, Groq…)
  anthropic_adapter.py — Anthropic Claude API (Messages format)
  local_adapter.py     — Ollama local model server

All adapters are registered in ProviderRouter (gateway/router.py) and selected
at request time based on the routing decision.
"""
