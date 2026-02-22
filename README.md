# Covenant

> A self-hostable AI gateway that hardens every LLM request before it reaches the model and every response before it reaches the caller.

Drop-in OpenAI-compatible API (`/v1/chat/completions`). Sits in front of OpenAI, Anthropic, or your own Ollama instance. Adds a security and reliability layer with zero code changes to your application.

---

## Benchmark — Prompt Injection Detection

Measured on 406 samples (203 injection, 203 clean). Dataset: [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) + [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), 50/50 balance, seed 42. Latency: sequential single-sample CPU inference (Apple M-series), 50-sample warmup, per-sample `time.perf_counter()`.

| Model | Precision | Recall | F1 | p50 ms | p99 ms | RPS |
| --- | --- | --- | --- | --- | --- | --- |
| ProtectAI DeBERTa-v3 (184M) | **1.0000** | **0.4286** | **0.6000** | 72.8 | 187.7 | 12.6 |
| Meta PromptGuard-2 86M | **1.0000** | 0.2463 | 0.3953 | 73.5 | 128.9 | 13.0 |
| Meta PromptGuard-2 22M | **1.0000** | 0.2118 | 0.3496 | **32.3** | **125.9** | **25.9** |

**What these numbers mean:**

- **Precision 1.00** — zero false positives. Not a single clean Alpaca prompt was flagged across 203 samples and every threshold tested (0.05–0.99). The model only fires when it's certain.
- **Recall 0.43 / 0.25 / 0.21** — all three models are conservative. The injections they miss are indirect/subtle patterns (context manipulation, goal hijacking) rather than direct "ignore previous instructions" attacks — those are caught by Tier 1 regex before ML is ever called. ProtectAI DeBERTa catches 2x more than Meta's models on this dataset.
- **The two tiers complement each other.** Tier 1 regex handles keyword-based attacks in < 1ms and short-circuits before any ML model is called. Tier 2 ML handles ambiguous paraphrased attacks with no obvious keywords.
- **ProtectAI DeBERTa-v3 is the best overall.** Same precision as both Meta models, highest recall (43% vs 25%/21%), and similar p50 latency. Meta PG2-22M is 2.3× faster but catches half as many injections — worth considering only when latency is the hard constraint.

**Methodology:** These are sequential single-sample numbers — the benchmark calls one sample at a time in a loop, not concurrently. In production, `run_in_executor` dispatches each inference call to a thread pool, so multiple requests can be in-flight simultaneously (per-request latency stays the same; throughput scales with CPU cores and thread pool size). To reduce per-request latency: a GPU brings Tier 2 to ~3ms; ONNX export + int8 quantization reaches ~15–20ms on CPU.

---

## Architecture

Every request passes through a 10-stage pipeline:

```
Client
  │
  ▼
NGINX (port 80)
  │  30 req/s per IP, burst 20  ·  security headers  ·  JSON logs
  ▼
FastAPI Gateway (port 8000)
  │
  ├─ 1.  Rate Limiter         Redis sliding-window, per API key (Lua atomic)
  ├─ 2.  Pattern Guard        Regex Tier 1 — CRITICAL/HIGH short-circuit (<1ms)
  ├─ 3.  Cache Lookup         FAISS ANN → cosine threshold → Redis payload fetch
  ├─ 4.  ML Guard             DeBERTa-v3 Tier 2 — ambiguous prompts only (~100ms)
  ├─ 5.  Pre-call Contracts   Behavioral rules checked on the request
  ├─ 6.  Provider Router      Alias → COST / LATENCY / EXPLICIT policy
  ├─ 7.  Provider Adapter     OpenAI / Anthropic / Ollama HTTP call
  ├─ 8.  Post-call Contracts  Rules checked on the response
  ├─ 9.  Cache Write          Embed + store if cacheable
  └─ 10. Langfuse Trace       Full span with per-stage metadata
```

---

## Features

### Two-Tier Prompt Injection Detection

**Tier 1 — Regex Pattern Guard** (`< 1ms`)

Patterns in [`gateway/security/patterns.json`](gateway/security/patterns.json) — hot-reloadable without restart. Three severity levels:

- `CRITICAL` / `HIGH` — short-circuit the pipeline immediately. The ML model is never called.
- `MEDIUM` — flagged for ML scoring.

Covers: `ignore previous instructions`, role-play jailbreaks (`you are now DAN`), system-prompt exfiltration, indirect injection phrases, prompt boundary attacks.

**Tier 2 — ML Guard** (`~100ms CPU`)

[`ProtectAI/deberta-v3-base-prompt-injection-v2`](https://huggingface.co/ProtectAI/deberta-v3-base-prompt-injection-v2) (184M params). Only runs when Tier 1 passes — ambiguous prompts that have no obvious keywords.

- DeBERTa-v3 uses disentangled attention (separate position + content matrices), giving stronger contextual understanding than DistilBERT for subtle/paraphrased attacks.
- PyTorch runs in `loop.run_in_executor` — never blocks the async event loop.
- Device auto-selection: CUDA → MPS (Apple Silicon) → CPU.
- Graceful degradation: if model not downloaded, passes with a warning log rather than crashing.

---

### Semantic Response Cache

Avoids redundant LLM calls for semantically equivalent questions ("What is Python?" vs "Can you explain Python?").

- **Embedding**: `all-MiniLM-L6-v2` — 384-dim float32 vectors, L2-normalized.
- **Search**: FAISS `IndexIDMap2(IndexFlatIP)` — inner product on normalized vectors = cosine similarity.
- **Threshold**: configurable (default 0.92 cosine similarity).
- **Storage**: vector in FAISS (in-process), payload in Redis with TTL. Separation allows Redis TTL to handle expiry natively.
- **Persistence**: binary FAISS index + JSON sidecar survives restarts.
- **Deletion**: `IndexIDMap2` supports `remove_ids` — invalidation and orphan cleanup are both handled.

---

### Behavioral Contracts

Per-application declarative rules that Covenant enforces on every request and response. Defined in JSON, hot-reloaded without restart.

**9 contract types across 3 evaluation tiers:**

| Tier | Latency | Types |
|------|---------|-------|
| Deterministic | < 1ms | `keyword`, `regex`, `length_limit`, `language_match`, `json_schema` |
| Classifier | ~10-15ms | `sentiment`, `topic_boundary` (zero-shot NLI via BART) |
| LLM Judge | ~100-300ms | `llm_judge` (natural language assertion via a small judge model) |

**Execution model:**
- `BLOCK` contracts run synchronously in parallel — response waits.
- `FLAG` / `LOG` contracts fire as background `asyncio.Task` — response returns immediately.
- LLM judge contracts should always be `FLAG` (never block the user for 200ms).

**Drift detection:** compliance scores (0.0–1.0) stored in Redis sorted sets per `(app_id, contract_id)`. Rolling 24h vs 7d baseline. `DriftAlert` fires when compliance drops > 10% relative to baseline. Dashboard endpoint: `GET /v1/contracts/{app_id}/drift`.

Example contract file ([`contracts/bank-support-bot.json`](contracts/bank-support-bot.json)):

```json
{
  "app_id": "bank-support-bot",
  "contracts": [
    {
      "type": "topic_boundary",
      "action": "BLOCK",
      "allowed_topics": ["banking", "finance", "account support"],
      "threshold": 0.7
    },
    {
      "type": "length_limit",
      "action": "FLAG",
      "max_words": 500
    },
    {
      "type": "llm_judge",
      "action": "LOG",
      "assertion": "The response does not give specific financial advice or recommend specific investments."
    }
  ]
}
```

---

### Provider Routing

Unified API across providers. Your application sends a single request format — Covenant routes it.

**Supported providers:**
- OpenAI (`gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`, `o1-mini`)
- Anthropic (`claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`, `claude-3-5-sonnet`, `claude-3-haiku`)
- Ollama local (`llama3.2`, `llama3.1`, `mistral`, `mixtral`, `phi3`, `gemma2`, `qwen2.5`, `codellama`)

**Routing policies:**
- `COST` — prefers the cheaper model tier for the target provider
- `LATENCY` — prefers the model with the lowest measured EMA latency (α = 0.2, updated after every call)
- `EXPLICIT` — respects the `X-Provider` request header

---

### Rate Limiting

Two layers:

1. **NGINX** (`limit_req_zone`) — 30 req/s per IP, burst 20. Network-level. Python process never wakes for burst traffic.
2. **Redis** (sliding-window) — per API key. Atomic via Lua script (single round-trip). Configurable window and limit.

---

### Observability

- **Langfuse distributed traces** — `GatewayTrace` wraps each request; `span()` creates a child span for each of the 10 pipeline stages. Falls back gracefully if Langfuse is unavailable.
- **Prometheus metrics** — counters: requests, injections blocked, cache hits/misses. Histograms: end-to-end latency, ML inference latency. `/metrics` restricted to RFC1918 subnets in NGINX.

---

## Getting Started

### Prerequisites

- Docker + Docker Compose
- An API key for at least one provider (OpenAI, Anthropic), or Ollama running locally

### Quickstart (Docker Compose)

```bash
# 1. Clone and configure
git clone https://github.com/Vinesh2929/Covenant.git
cd Covenant
cp .env.example .env
# Edit .env — add OPENAI_API_KEY and/or ANTHROPIC_API_KEY

# 2. Download models (first time only — ~500MB)
python scripts/download_models.py

# 3. Start the full stack
docker compose up --build
```

Gateway is now available at `http://localhost/v1/chat/completions`.
Langfuse dashboard at `http://localhost:3000`.

### Send a request

```bash
curl http://localhost/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Local development (no Docker)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Redis
redis-server

# 3. Download models
python scripts/download_models.py

# 4. Configure environment
cp .env.example .env  # fill in API keys

# 5. Start gateway
uvicorn gateway.main:app --reload --port 8000
```

---

## Configuration

All configuration via environment variables. Copy `.env.example` to `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_API_KEY` | required | Bearer token callers must provide |
| `OPENAI_API_KEY` | — | OpenAI provider key |
| `ANTHROPIC_API_KEY` | — | Anthropic provider key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `ML_GUARD_THRESHOLD` | `0.5` | Injection confidence threshold (0–1) |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity threshold for cache hits |
| `CACHE_TTL_SECONDS` | `3600` | Redis TTL for cached responses |
| `RATE_LIMIT_REQUESTS` | `100` | Requests allowed per window per API key |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding window size |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse project public key (optional) |
| `LANGFUSE_SECRET_KEY` | — | Langfuse project secret key (optional) |
| `ROUTING_POLICY` | `COST` | `COST` \| `LATENCY` \| `EXPLICIT` |

---

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible. All standard fields work. Gateway-specific extensions:

```json
{
  "model": "gpt-4o",
  "messages": [...],
  "app_id": "my-app",         // optional — enables per-app behavioral contracts
  "x_provider": "anthropic"   // optional — forces provider (EXPLICIT routing)
}
```

Returns standard OpenAI response format regardless of the underlying provider.

**Error responses:**

| Status | Meaning |
|--------|---------|
| `400` | Prompt injection detected (Tier 1 or Tier 2) |
| `422` | Behavioral contract violation (BLOCK action) |
| `429` | Rate limit exceeded |
| `502` | Upstream provider error after retries |

### `GET /health`

Returns `{"status": "ok"}`. Served by NGINX directly — no Python process required.

### `GET /metrics`

Prometheus metrics. Restricted to RFC1918 subnets (not publicly accessible by default).

### `GET /v1/contracts/{app_id}`

List all active contracts for an application.

### `GET /v1/contracts/{app_id}/drift`

Current drift status — recent compliance scores vs 7-day baseline, any active `DriftAlert`s.

### `POST /v1/contracts/{app_id}/reload`

Hot-reload contracts from disk for an application without restarting the gateway.

---

## Development

### Project Structure

```
gateway/
  main.py                   FastAPI app + 10-stage pipeline
  config.py                 Pydantic settings (all env vars)
  rate_limiter.py           Redis sliding-window rate limiter
  router.py                 Provider routing + model aliases
  security/
    guard.py                SecurityGuard — orchestrates Tier 1 + 2
    pattern_guard.py        Tier-1 regex scanner
    patterns.json           Curated injection patterns (hot-reloadable)
    ml_guard.py             Tier-2 DeBERTa-v3 classifier
  cache/
    embedder.py             SentenceTransformer wrapper
    store.py                FAISS index with persistence
    semantic_cache.py       Cache orchestrator
  contracts/
    schema.py               Contract DSL — 9 types, 3 tiers
    evaluator.py            Per-type evaluators with compliance scoring
    registry.py             Tiered execution engine (sync BLOCK + async FLAG)
    drift.py                Redis-backed drift detection
  observability/
    langfuse_client.py      Distributed tracing
    metrics.py              Prometheus metrics
  providers/
    base.py                 BaseProviderAdapter + retry logic
    openai_adapter.py
    anthropic_adapter.py
    local_adapter.py        Ollama NDJSON streaming
scripts/
  download_models.py        Download models before first start
  build_benchmark_dataset.py  Build labeled eval dataset
  benchmark_security.py    Three-model injection benchmark
```

### Running Tests

```bash
pip install -r requirements-dev.txt  # includes lupa for fakeredis Lua support
pytest tests/ -v
# 63/63 pass — no real Redis, no models, no API keys required
```

### Running the Benchmark

```bash
# Build dataset (one-time, ~406 samples from HuggingFace)
python scripts/build_benchmark_dataset.py

# Benchmark ProtectAI DeBERTa (no login needed)
python scripts/benchmark_security.py --dataset data/benchmark_dataset.jsonl --pr-curve

# Also benchmark Meta PromptGuard 86M + 22M (requires HF login + Llama license)
huggingface-cli login
python scripts/benchmark_security.py --dataset data/benchmark_dataset.jsonl --all-models --pr-curve
```

Results written to `results/benchmark_results.json`. PR curve PNG at `results/pr_curve.png`.

---

## Deployment Notes

### GPU Acceleration

Set `device=0` in `gateway/security/ml_guard.py` to run the ML guard on GPU. On Apple Silicon, MPS is selected automatically. Expected latency improvement: ~5-10x over CPU.

### Horizontal Scaling

Multiple gateway instances can share a single Redis — rate limiting and cache payloads are consistent. Each instance maintains its own FAISS index (trade-off: cache warming per instance on cold start). For shared FAISS, deploy a vector database instead of in-process FAISS.

### TLS

Terminate TLS at NGINX. Add `ssl_certificate` and `ssl_certificate_key` directives to `nginx/nginx.conf` and update the upstream block.

---

## License

MIT
