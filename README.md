# Covenant

A self-hostable AI gateway that hardens every LLM request before it reaches the model and every response before it reaches the caller.

Drop-in OpenAI-compatible API (`/v1/chat/completions`). Sits in front of OpenAI, Anthropic, or your own Ollama instance. Zero code changes to your application.

Blog: [vineshnathan.ca/covenant](https://www.vineshnathan.ca/covenant)
---

## Benchmark — Prompt Injection Detection

406 samples (203 injection, 203 clean). Dataset: [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) + [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca). CPU inference, 50-sample warmup, per-sample `time.perf_counter()`.

| Model | Precision | Recall | F1 | p50 ms | p99 ms |
| --- | --- | --- | --- | --- | --- |
| ProtectAI DeBERTa-v3 (184M) | **1.0000** | 0.4286 | 0.6000 | 72.8 | 187.7 |
| Meta PromptGuard-2 86M | **1.0000** | 0.2463 | 0.3953 | 73.5 | 128.9 |
| Meta PromptGuard-2 22M | **1.0000** | 0.2118 | 0.3496 | 32.3 | 125.9 |

| Pipeline | Recall | Notes |
| --- | --- | --- |
| Tier 2 alone | 42.9% | Zero false positives |
| **Tier 2 + Tier 3** | **63.5%** | +20.7pp, async, ~$0.05/1k requests |

Tier 3 (Claude Haiku, async) ran on all 116 Tier-2 misses and caught 42 — multilingual attacks, obfuscated injections, persona hijacking that the ML classifier scores as benign. Zero added latency: the judge fires after the response is already sent.

---

## Architecture

Every request passes through a 10-stage pipeline:

```
Client → NGINX → FastAPI Gateway
  1.  Rate Limiter         Redis sliding-window, per API key (Lua atomic)
  2.  Pattern Guard        Regex Tier 1 — CRITICAL/HIGH short-circuit (<1ms)
  3.  Cache Lookup         FAISS ANN + cosine threshold + Redis payload fetch
  4.  ML Guard             DeBERTa-v3 Tier 2 — ambiguous prompts only (~100ms)
  5.  Pre-call Contracts   Behavioral rules checked on the request
  6.  Provider Router      Alias → COST / LATENCY / EXPLICIT policy
  7.  Provider Adapter     OpenAI / Anthropic / Ollama HTTP call
  8.  Post-call Contracts  Rules checked on the response
  9.  Cache Write          Embed + store if cacheable
  10. Langfuse Trace       Full span with per-stage metadata
```

---

## Features

### Three-tier prompt injection detection

Regex (<1ms) → DeBERTa-v3 (~100ms) → LLM judge (async, never blocks the caller). Each tier handles what the previous can't. Combined recall 63.5%, zero false positives.

### Semantic response cache

FAISS ANN (all-MiniLM-L6-v2, 384-dim) + Redis payload store with TTL. Default 0.92 cosine similarity threshold. Index survives restarts.

### Behavioral contracts

Per-app declarative rules on every request and response. 9 contract types across 3 tiers: deterministic (<1ms), classifier (~15ms), LLM judge (~200ms). `BLOCK` runs synchronously; `FLAG`/`LOG` fire as background tasks. Drift detection via Redis time-series with 7-day rolling baseline.

### Provider routing

OpenAI, Anthropic, Ollama — unified request format, zero application changes. `COST`, `LATENCY`, or `EXPLICIT` routing policies.

### Rate limiting

NGINX per-IP (30 req/s, burst 20) + Redis per-API-key sliding window (Lua atomic, single round-trip).

### Observability

Langfuse distributed traces (10 child spans per request) + Prometheus metrics (`/metrics`, RFC1918 only).

---

## Getting Started

### Docker Compose

```bash
git clone https://github.com/Vinesh2929/Covenant.git && cd Covenant
cp .env.example .env          # fill in your API keys
python scripts/download_models.py
docker compose up --build
```

Gateway: `http://localhost/v1/chat/completions` · Langfuse: `http://localhost:3000`

### Local (no Docker)

```bash
pip install -r requirements.txt
redis-server &
cp .env.example .env
uvicorn gateway.main:app --reload --port 8000
```

---

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `GATEWAY_API_KEY` | required | Bearer token callers must provide |
| `OPENAI_API_KEY` | — | OpenAI provider key |
| `ANTHROPIC_API_KEY` | — | Anthropic provider key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `ML_GUARD_THRESHOLD` | `0.5` | Injection confidence threshold (0–1) |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Cache hit threshold |
| `ROUTING_POLICY` | `COST` | `COST` \| `LATENCY` \| `EXPLICIT` |

---

## API

`POST /v1/chat/completions` — OpenAI-compatible. Add `"app_id": "my-app"` to enable behavioral contracts.

| Status | Meaning |
| --- | --- |
| `400` | Prompt injection detected |
| `422` | Contract violation (`BLOCK`) |
| `429` | Rate limit exceeded |
| `502` | Upstream provider error |

`GET /health` · `GET /metrics` · `GET /v1/contracts/{app_id}` · `GET /v1/contracts/{app_id}/drift` · `POST /v1/contracts/{app_id}/reload`

---

## Development

### Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v
# 104/104 pass — no external services required
```

### Benchmark

```bash
python scripts/build_benchmark_dataset.py
PYTHONPATH=. python scripts/benchmark_security.py --dataset data/benchmark_dataset.jsonl
PYTHONPATH=. python scripts/benchmark_security.py --dataset data/benchmark_dataset.jsonl --validate-tier3
```

### Project structure

```
gateway/
  main.py               FastAPI app + 10-stage pipeline
  config.py             Pydantic settings
  rate_limiter.py       Redis sliding-window rate limiter
  router.py             Provider routing + model aliases
  security/
    guard.py            SecurityGuard — Tier 1 + 2 + 3 orchestrator
    pattern_guard.py    Tier 1 regex scanner
    ml_guard.py         Tier 2 DeBERTa-v3 classifier
    llm_guard.py        Tier 3 async LLM judge
    patterns.json       Injection patterns (hot-reloadable)
  cache/
    embedder.py         SentenceTransformer wrapper
    store.py            FAISS index with persistence
    semantic_cache.py   Cache orchestrator
  contracts/
    schema.py           Contract DSL — 9 types, 3 tiers
    evaluator.py        Per-type evaluators
    registry.py         Tiered execution engine
    drift.py            Redis-backed drift detection
  observability/
    langfuse_client.py  Distributed tracing
    metrics.py          Prometheus metrics
  providers/
    openai_adapter.py
    anthropic_adapter.py
    local_adapter.py    Ollama NDJSON streaming
```

---

## License

MIT
