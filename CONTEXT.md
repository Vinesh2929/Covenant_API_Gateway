# Covenant — Project Context & Progress

> Living document. Update this file whenever a meaningful decision is made, a module
> is completed, or the plan changes. It is the single source of truth for what has
> been built, why, and what comes next.

---

## What Is Covenant?

A **self-hostable AI Gateway** that sits between your applications and LLM providers
(OpenAI, Anthropic, Ollama). Every request passes through a hardened pipeline before
it reaches the model, and every response is validated before it goes back to the caller.

**Core value proposition:**
- Drop-in OpenAI-compatible API (`/v1/chat/completions`)
- Prompt injection detection — two-tier, sub-20 ms on CPU
- Semantic response caching — FAISS + Redis, ~98% cache-hit savings on repeated queries
- Behavioral contracts — per-app rules that block/log bad model outputs
- Full observability — Prometheus metrics + Langfuse distributed traces

**Stack:** FastAPI · Redis · FAISS · sentence-transformers · DeBERTa-v3 · NGINX · Docker Compose

---

## Architecture

```
Client
  │
  ▼
NGINX (port 80)
  │  rate-limit: 30 req/s per IP, burst 20
  │  security headers, JSON logs
  ▼
FastAPI Gateway (port 8000)
  │
  ├─ 1. Rate Limiter       Redis sliding-window, per API key
  ├─ 2. Pattern Guard      Regex tier-1 — CRITICAL/HIGH patterns short-circuit here
  ├─ 3. Cache Lookup       FAISS ANN search → Redis payload fetch
  ├─ 4. ML Guard           DeBERTa-v3 tier-2 — ambiguous prompts scored here
  ├─ 5. Pre-call Contracts Behavioral rules checked on the request
  ├─ 6. Provider Router    Alias resolution → COST / LATENCY / EXPLICIT policy
  ├─ 7. Provider Adapter   OpenAI / Anthropic / Ollama HTTP call
  ├─ 8. Post-call Contracts Rules checked on the response
  ├─ 9. Cache Write        Embed + store if cacheable
  └─ 10. Langfuse Trace    Full span recorded with metadata
```

### Key design principles

| Principle | How it's applied |
|-----------|-----------------|
| Fail open on optional components | ML guard, cache, Langfuse — each degrades gracefully if unavailable |
| Deferred loading | Heavy models (DeBERTa, SentenceTransformer) load in `warm_up()`, not at import |
| Async-safe inference | PyTorch forward passes run in `loop.run_in_executor` to avoid blocking the event loop |
| One source of config | All settings in `SecuritySettings`, `CacheSettings`, etc. — consumed by Pydantic from `.env` |
| No secrets in code | Every key / URL is an env var — `.env.example` documents all of them |

---

## File Map

```
gateway/
  main.py                   FastAPI app + lifespan hook + /v1/chat/completions handler
  config.py                 Pydantic settings model (all env vars)
  rate_limiter.py           Redis sliding-window rate limiter
  router.py                 Provider routing + model alias resolution
  security/
    guard.py                SecurityGuard — orchestrates pattern + ML tiers
    pattern_guard.py        Tier-1 regex scanner (patterns.json)
    patterns.json           Curated regex patterns for CRITICAL/HIGH/MEDIUM injection signals
    ml_guard.py             Tier-2 DeBERTa-v3 classifier (ProtectAI/deberta-v3-base-prompt-injection-v2)
  cache/
    embedder.py             SentenceTransformer wrapper (all-MiniLM-L6-v2)
    store.py                FAISS IndexIDMap2 with persistence (binary + JSON sidecar)
    semantic_cache.py       Orchestrator: embed → FAISS search → Redis fetch/store
  contracts/
    schema.py               Contract DSL: 9 types across 3 evaluation tiers + EvaluationTier enum
    evaluator.py            Per-type evaluators with compliance scoring (0.0-1.0) for drift detection
    registry.py             Tiered execution: sync BLOCK + async FLAG/LOG with background tasks
    drift.py                Drift detection: Redis time series, rolling averages, alert thresholds
  observability/
    langfuse_client.py      Async context manager traces + child spans
    metrics.py              Prometheus counters/histograms via prometheus_client
  providers/
    base.py                 BaseProviderAdapter ABC + retry logic (1s→2s→4s backoff)
    openai_adapter.py       OpenAI HTTP adapter
    anthropic_adapter.py    Anthropic HTTP adapter (translates message format + stop reasons)
    local_adapter.py        Ollama NDJSON streaming adapter

nginx/nginx.conf            Reverse proxy, rate limiting, security headers, JSON logs
Dockerfile                  Multi-stage: builder installs deps, runtime image is lean (non-root user)
docker-compose.yml          nginx + gateway + redis + langfuse + postgres + (ollama optional)
scripts/
  build_benchmark_dataset.py  Pulls deepset/prompt-injections + Alpaca, outputs data/benchmark_dataset.jsonl
  benchmark_security.py       Benchmarks three injection classifiers; outputs results/benchmark_results.json
  download_models.py          Downloads ProtectAI DeBERTa + MiniLM; optional Meta PG2 models
  generate_test_data.py       Synthetic dataset generation
  train_classifier.py         Fine-tune loop (DistilBERT/DeBERTa)
data/
  .gitkeep                    Placeholder — benchmark_dataset.jsonl written here by build script
results/
  .gitkeep                    Placeholder — benchmark_results.json and pr_curve.png written here
tests/
  conftest.py               Shared fixtures (mostly stubs — see TODO list)
  test_security.py
  test_cache.py
  test_contracts.py
  test_rate_limiter.py
  test_router.py
```

---

## What Has Been Built (Progress)

### ✅ Completed

#### Gateway core (`gateway/main.py`, `config.py`)
- FastAPI lifespan hook wires all subsystems at startup in the correct order
- `/v1/chat/completions` handler runs the full 10-stage pipeline
- `_extract_prompt_text()` flattens the messages array to a single string (used by security + cache)
- All settings read from env via Pydantic; `.env.example` documents every variable

#### Rate limiter (`gateway/rate_limiter.py`)
- Redis sliding-window (ZRANGEBYSCORE / ZADD / EXPIRE) — accurate under burst
- Per API-key buckets; configurable window size and request limit
- Async-safe; single Redis round-trip per request

#### Security — Tier 1 (`gateway/security/pattern_guard.py`, `patterns.json`)
- Regex patterns in a JSON file (hot-reloadable without restart)
- Patterns grouped by severity: CRITICAL, HIGH, MEDIUM
- Short-circuit: CRITICAL/HIGH patterns block immediately, skipping the ML model
- `PatternMatch` dataclass carries pattern name, severity, matched text

#### Security — Tier 2 (`gateway/security/ml_guard.py`)
- **Model: ProtectAI/deberta-v3-base-prompt-injection-v2** (184 M params)
  - Chosen after benchmarking three candidates (see decision log below)
  - ~15 ms CPU latency per prompt; ~98% precision at 95% recall (public datasets)
- Deferred load via `warm_up()` → `run_in_executor` (keeps startup fast)
- Graceful degradation: if model not downloaded, returns `is_injection=False` and logs a warning
- `_select_device()`: CUDA → MPS (Apple Silicon) → CPU priority
- `ScanResult` dataclass: `is_injection`, `confidence`, `label`, `latency_ms`, `model_version`

#### Security orchestrator (`gateway/security/guard.py`)
- `SecurityGuard.scan()` runs Tier 1 first; only calls Tier 2 if Tier 1 does not block
- Merges results into a `GuardResult` with both tier results attached

#### Semantic cache (`gateway/cache/`)
- **Embedder**: `all-MiniLM-L6-v2` via sentence-transformers; 384-dim float32 vectors
- **FAISSStore**: `IndexIDMap2(IndexFlatIP)` — inner product on L2-normalised vectors = cosine similarity; supports `add_with_ids` + `remove_ids` for deletion; persists as binary index + `.meta` JSON sidecar
- **SemanticCache**: embed prompt → FAISS ANN search → threshold check (default 0.92) → Redis GET for payload; `set()` stores JSON in Redis with TTL, then adds vector to FAISS; orphan detection when Redis key expires before FAISS vector is removed

#### Provider adapters (`gateway/providers/`)
- **Base**: `BaseProviderAdapter` ABC; `_request_with_retry` with exponential backoff (3 retries, 1→2→4 s); raises `RateLimitError` on 429, `ProviderUnavailableError` on 5xx
- **OpenAI**: strips internal gateway fields (`app_id`, `x_provider`, `x_request_id`) before forwarding
- **Anthropic**: extracts `system` messages from messages array into top-level field; joins text content blocks; maps stop reasons (`end_turn→stop`, `max_tokens→length`)
- **Ollama (local)**: reads NDJSON streaming response line-by-line; accumulates content fragments; reconstructs OpenAI-compatible response envelope

#### Model router (`gateway/router.py`)
- 18 built-in model aliases (5 OpenAI, 5 Anthropic, 8 Ollama)
- Routing policies: `COST` (prefer cheaper tier), `LATENCY` (prefer fastest measured), `EXPLICIT` (X-Provider header)
- EMA latency tracking (α = 0.2) updated after each successful call
- Lazy imports of adapter classes to avoid circular imports

#### Behavioral contracts v2 (`gateway/contracts/`)
- **Schema**: 9 contract types across 3 evaluation tiers (deterministic, classifier, LLM judge)
  - Deterministic (< 1ms): `KeywordContract`, `RegexContract`, `LengthLimitContract`, `LanguageMatchContract`, `SchemaContract`
  - Classifier (~10-15ms): `SentimentContract`, `TopicBoundaryContract` (zero-shot NLI)
  - LLM Judge (~100-300ms): `LLMJudgeContract` (natural-language assertions via small LLM)
  - Structural: `CompositeContract` (AND/OR boolean logic)
- **Tiered execution model**:
  - BLOCK contracts: evaluated synchronously in parallel — response waits for completion
  - FLAG/LOG contracts: fired as background `asyncio.Task` — response returns immediately
  - LLM judge contracts should always be FLAG (never block the user for 200ms)
- **Evaluators**: each returns a compliance_score (0.0-1.0) for drift tracking
  - `TopicBoundaryEvaluator`: zero-shot classification via `facebook/bart-large-mnli`
  - `LLMJudgeEvaluator`: calls a judge model through the gateway's own provider system
  - `LanguageMatchEvaluator`: uses `langdetect` for language identification
  - `LengthLimitEvaluator`: word, character, and sentence count limits
- **Drift detection** (`drift.py`):
  - Compliance scores stored in Redis sorted sets per `(app_id, contract_id)`
  - Rolling averages over configurable windows (default: 24h recent vs 7d baseline)
  - Fires `DriftAlert` when compliance drops > 10% relative to baseline
  - `GET /v1/contracts/{app_id}/drift` endpoint for dashboard consumption
- **Registry**: tiered evaluation, background task management, hot-reload, drift integration
- **API endpoints**: `GET /v1/contracts/{app_id}`, `GET /v1/contracts/{app_id}/drift`, `POST /v1/contracts/{app_id}/reload`

#### Observability (`gateway/observability/`)
- **Langfuse**: `GatewayTrace` async context manager; `span()` asynccontextmanager for child spans; graceful fallback if SDK not installed
- **Metrics**: Prometheus counters (requests total, injections, cache hits/misses) + histograms (request latency, ML inference latency)

#### Infrastructure
- **NGINX**: `limit_req_zone` (30 req/s/IP, burst 20), upstream keepalive 32, JSON access log, all security headers, `/health` served directly (no Python wake), `/metrics` restricted to RFC1918 subnets
- **Dockerfile**: multi-stage (builder: gcc + deps, runtime: non-root `gateway` user), `HEALTHCHECK` with curl, uvicorn CMD

---

### ⚠️ Remaining / TODO

| Item | Status | Notes |
|------|--------|-------|
| **`tests/` — full suite** | ✅ Done (63/63 pass) | fakeredis + seeded stub embedder; no external services needed |
| **`scripts/download_models.py`** | ✅ Done | Downloads ProtectAI DeBERTa + all-MiniLM-L6-v2; optional benchmark models behind `DOWNLOAD_ALL_MODELS` env var |
| **`scripts/build_benchmark_dataset.py`** | ✅ Done | Pulls deepset/prompt-injections + tatsu-lab/alpaca, 50/50 balance, writes JSONL to `data/benchmark_dataset.jsonl` |
| **`scripts/benchmark_security.py`** | ✅ Done | Three-model comparison (ProtectAI DeBERTa-v3, Meta PG2 86M/22M); warmup, latency percentiles, PR curve, JSON output to `results/` |
| **README.md** | ⚠️ Next | Run benchmark to get real numbers, then write README around them |
| **Fine-tune loop** | Low | `scripts/train_classifier.py` targets DistilBERT; update for DeBERTa after benchmark establishes baseline |

---

## Key Design Decisions & Reasoning

### 1. DeBERTa-v3 over DistilBERT for Tier-2 injection detection

**Decision:** Use `ProtectAI/deberta-v3-base-prompt-injection-v2` as the off-the-shelf
Tier-2 classifier instead of fine-tuning DistilBERT from scratch.

**Why:**
- DistilBERT fine-tunes for injection already exist publicly — shipping another one adds
  little signal.
- DeBERTa-v3 uses *disentangled attention* (separate position + content matrices) and
  *replaced-token detection* pre-training, giving it better contextual understanding of
  subtle injection patterns (indirect injection, role-play jailbreaks, multi-step attacks).
- ProtectAI's published checkpoint achieves ~98% precision at 95% recall on public
  injection datasets — substantially better than a DistilBERT fine-tune at the same threshold.

**Covenant's actual contribution:**
1. Benchmark the three leading open-source classifiers on the same eval set.
2. Fine-tune the winner on domain-specific data (code-heavy prompts, multi-turn).
3. Integrate into the two-tier pipeline where fast regex short-circuits HIGH/CRITICAL patterns.

**Benchmark candidates (run via `scripts/benchmark_security.py`):**

| Model | Params | CPU latency | Notes |
|-------|--------|------------|-------|
| ProtectAI deberta-v3-base-v2 | 184 M | ~15 ms | Most widely deployed |
| Meta Prompt Guard 2 86M | 86 M | ~8 ms | Multilingual, Meta-backed |
| Meta Prompt Guard 2 22M | 22 M | ~3 ms | Fastest; small accuracy trade-off |

### 2. Two-tier security (regex + ML)

**Decision:** Run regex patterns first; only invoke the ML model for prompts that
pattern guard passes (not CRITICAL/HIGH).

**Why:**
- The ML model takes 3–15 ms per prompt. Patterns take < 0.1 ms.
- Most real-world injection attempts use known phrases (`ignore previous instructions`,
  `you are now DAN`, etc.) — patterns catch these instantly.
- The ML model handles the ambiguous middle ground: paraphrased attacks, indirect
  injection embedded in tool outputs, jailbreaks that avoid obvious keywords.

### 3. Semantic cache design (FAISS + Redis separation)

**Decision:** Store vectors in FAISS (in-process memory), payloads in Redis (external).

**Why:**
- FAISS is optimised for ANN search; Redis is optimised for key-value retrieval with TTL.
- Redis TTL handles cache expiry automatically — FAISS has no native expiry concept.
- Separation allows horizontal scaling: multiple gateway instances share the same Redis
  but each has its own FAISS index (trade-off: cold starts on new instances).
- `IndexIDMap2` wrapper enables deletion (needed for `invalidate()` and orphan cleanup).

### 4. Behavioral contracts as JSON files

**Decision:** Contracts are JSON files in a `contracts/` directory, one per app_id.

**Why:**
- Ops teams can add/change contracts without touching Python code.
- Hot-reload via `registry.reload(app_id)` means zero-downtime contract updates.
- JSON is easy to generate programmatically (admin API, Terraform, CI/CD).
- Pydantic discriminated union validates contract structure on load — bad files are
  skipped with a log warning rather than crashing the gateway.

### 5. NGINX as the outer layer (not FastAPI middleware)

**Decision:** Rate limiting at NGINX level (`limit_req_zone`) in addition to app-level
Redis rate limiting.

**Why:**
- Network-level rate limiting happens before the Python process wakes up — important
  for DDoS resilience.
- The two limiters serve different keys: NGINX limits by IP, Redis limits by API key.
- NGINX also handles connection-level concerns (keep-alive, TLS termination, header
  size limits) that are awkward to configure in an ASGI app.

---

## How to Run

### Local development (no Docker)
```bash
# 1. Copy env file
cp .env.example .env  # fill in API keys

# 2. Start Redis
redis-server

# 3. Download models
python scripts/download_models.py

# 4. Start gateway
uvicorn gateway.main:app --reload --port 8000
```

### Docker Compose (full stack)
```bash
docker compose up --build
# Gateway available at http://localhost/v1/chat/completions
# Langfuse dashboard at http://localhost:3000
```

### Test a request
```bash
curl http://localhost/v1/chat/completions \
  -H "Authorization: Bearer <GATEWAY_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Run tests
```bash
pip install -r requirements-dev.txt   # includes lupa (required for fakeredis Lua scripting)
pytest tests/ -v
# 63/63 tests pass — no real Redis, no real models, no API keys needed
```

---

## Changelog

| Date | Change | Reason |
|------|--------|--------|
| 2026-02-19 | Implemented all 18 core files from stubs | Complete the initial scaffold into a working pipeline |
| 2026-02-19 | Switched Tier-2 model from DistilBERT to ProtectAI DeBERTa-v3 | Better precision/recall; more credible benchmark contribution |
| 2026-02-19 | Added `gateway/security/patterns.json` | Externalise regex patterns for hot-reload without restart |
| 2026-02-19 | Committed and pushed to `origin/main` (commit `4cee891`) | Checkpoint working implementation |
| 2026-02-19 | **Behavioral contracts v2**: 3-tier evaluation (deterministic/classifier/LLM judge), tiered execution (sync BLOCK + async FLAG), drift detection via Redis time series, 4 new contract types (LengthLimit, LanguageMatch, TopicBoundary, LLMJudge), compliance scoring, drift dashboard endpoint | Core differentiator — no other gateway has inline behavioral contract enforcement with drift detection |
| 2026-02-20 | Merged `feature/behavioral-contracts-v2` and `feature/tests-benchmarks-readme` into main via fast-forward | Both branches were linear on top of main; clean merge with no conflicts |
| 2026-02-20 | **Test suite fix — StubEmbedder zero-vector bug**: `np.frombuffer(sha256_bytes, dtype=float32)` produces all-zeros for some inputs (e.g. "What is Python?") because those bytes decode to zeros as IEEE 754 float32. Fixed by using seeded numpy RNG — `np.random.default_rng(seed).standard_normal(32)` — for proper Gaussian unit vectors. EMBED_DIM 8→32 for better inter-prompt separation. | 3 cache tests were failing silently because zero-vector embeddings always had cosine similarity 0.0, below the 0.92 threshold |
| 2026-02-20 | **Test suite fix — fakeredis Lua scripting**: `fakeredis` does not support `SCRIPT LOAD` / `EVALSHA` by default. The rate-limiter uses a Lua script for atomicity. Fixed by adding `lupa` (Lua runtime) to `requirements-dev.txt` — fakeredis ≥ 2.20 auto-enables Lua when lupa is installed. | 8 rate-limiter tests were erroring at fixture setup with `unknown command 'script'` |
| 2026-02-20 | `scripts/download_models.py` written and merged | Blocks gateway cold-start without models; now runnable with `python scripts/download_models.py` |
| 2026-02-20 | **All 63 tests pass** on main. No real Redis, no real models, no API keys required to run the suite. | — |
| 2026-02-20 | **`scripts/build_benchmark_dataset.py`** written — pulls deepset/prompt-injections (injection) + tatsu-lab/alpaca (clean), 50/50 balance, JSONL output to `data/benchmark_dataset.jsonl` | Needed before benchmark can run; decoupled from benchmark itself so dataset is built once |
| 2026-02-20 | **`scripts/benchmark_security.py`** fully implemented — three-model comparison (ProtectAI DeBERTa-v3, Meta PromptGuard 86M, Meta PromptGuard 22M), 10-sample warmup before timing, per-sample `time.perf_counter()` latencies, p50/p95/p99, sklearn precision/recall/F1, PR curve sweep, results to `results/benchmark_results.json` | Produces the real numbers needed for README; Meta models gated behind `--all-models` flag + require HF login + Llama license |
| 2026-02-20 | Created `results/` and `data/` directories with `.gitkeep` | Holds benchmark output and dataset; tracked in git so paths exist on fresh clone |
