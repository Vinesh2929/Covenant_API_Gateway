"""
gateway/main.py

Entry point for the AI Gateway FastAPI application.

This file is the "conductor" of the entire system. It does not contain any
business logic itself — instead it wires all the subsystems together and
defines the order in which they run for every request.

Think of it like a router in a restaurant kitchen:
  - The waiter (NGINX) brings the order (HTTP request) to the pass.
  - The head chef (this file) decides which stations handle it and in what order.
  - Each station (rate limiter, security, cache, provider, contracts) does its
    job and passes control back.
  - The head chef assembles the final plate (response) and sends it out.
"""

from __future__ import annotations

import time        # for measuring how long each stage takes
import uuid        # for generating unique request IDs
from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis.asyncio as aioredis          # async Redis client
import structlog                           # structured JSON logging (better than print())
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gateway.config import Settings, get_settings
from gateway.rate_limiter import RateLimiter, RateLimitExceeded, WindowConfig
from gateway.router import ProviderRouter
from gateway.security.guard import SecurityGuard
from gateway.cache.semantic_cache import SemanticCache
from gateway.contracts.registry import ContractRegistry
from gateway.observability.langfuse_client import LangfuseClient
from gateway.observability.metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Module-level logger
#
# structlog wraps Python's standard logging but produces structured JSON output
# instead of free-form strings.  In production this makes logs searchable in
# tools like Datadog, Loki, or CloudWatch.
#
# Example output:
#   {"event": "request_complete", "provider": "openai", "latency_ms": 312.4, ...}
# ---------------------------------------------------------------------------
log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Helper: extract prompt text from the messages list
#
# The OpenAI chat format uses a "messages" array where each item has a "role"
# and "content".  We need a single string representation of the conversation
# for two purposes:
#   1. The SecurityGuard scans it for injection patterns.
#   2. The SemanticCache embeds it into a vector for similarity search.
#
# We include the role prefix ("user: ...", "assistant: ...") so the embedding
# captures the conversational structure, not just the raw text.
# ---------------------------------------------------------------------------

def _extract_prompt_text(body: dict) -> str:
    """
    Flatten the messages list into a single string.

    Why we need this:
      The SecurityGuard and SemanticCache both work on plain strings, but the
      OpenAI format sends messages as a list of objects.  We join them all so
      the full conversation context is considered, not just the last message.

    Handles multi-part content:
      The OpenAI API supports "content" as either a plain string OR a list of
      content blocks (for vision models that accept text + images).  We extract
      only the text blocks here.

    Args:
        body: The parsed JSON body of an incoming chat completion request.

    Returns:
        A single string: all messages joined by newlines, each prefixed with
        their role.  Example:
          "system: You are a helpful assistant.\nuser: What is 2+2?"
    """
    messages = body.get("messages", [])
    parts = []

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, str):
            # Simple case: content is just a plain string
            parts.append(f"{role}: {content}")

        elif isinstance(content, list):
            # Complex case: content is a list of typed blocks (text, image_url, etc.)
            # We only extract text blocks and ignore image references.
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(f"{role}: {block['text']}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helper: authenticate the inbound request
#
# Every request to the gateway must include an API key in the standard HTTP
# Authorization header:
#   Authorization: Bearer <your-gateway-api-key>
#
# This is the gateway's OWN key (set in .env as GATEWAY_API_KEY), completely
# separate from the upstream provider keys (OpenAI, Anthropic, etc.) which
# are stored server-side and never exposed to clients.
# ---------------------------------------------------------------------------

def _authenticate(request: Request, settings: Settings) -> str:
    """
    Validate the Authorization header and return the raw API key string.

    Why a separate function?
      Authentication is a cross-cutting concern.  By extracting it here, the
      proxy_request function stays focused on orchestration rather than auth
      details.  When you add JWT support or per-key rate limits later, you only
      touch this function.

    Args:
        request:  The incoming FastAPI Request object.
        settings: The root Settings object (contains gateway_api_key).

    Returns:
        The raw API key string (without the "Bearer " prefix).

    Raises:
        HTTPException 401: If the header is missing, malformed, or the key
                           does not match the configured gateway key.
    """
    auth_header = request.headers.get("Authorization", "")

    # Check the header is present and uses the Bearer scheme
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Authorization header missing or not using Bearer scheme.",
                    "type": "authentication_error",
                }
            },
        )

    # Strip the "Bearer " prefix to get the raw key
    api_key = auth_header.removeprefix("Bearer ").strip()

    # If no gateway key is configured (e.g. during local dev), skip key validation.
    # In production GATEWAY_API_KEY should always be set.
    if settings.gateway_api_key and api_key != settings.gateway_api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key.",
                    "type": "authentication_error",
                }
            },
        )

    return api_key


# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown
#
# FastAPI's lifespan hook is an async context manager that runs:
#   - BEFORE the server starts accepting requests (startup code)
#   - AFTER the server stops accepting requests (shutdown code)
#
# We use it to initialise all the "heavy" subsystems once at startup:
#   - Redis connection pool (avoids connecting on every request)
#   - ML model loading (DistilBERT is ~250 MB — we want it in memory, not cold-loaded)
#   - FAISS index loading (the semantic cache vector store)
#   - Contract registry scanning (reads contract files from disk)
#
# Everything is stored on `app.state` — FastAPI's built-in namespace for
# sharing objects across request handlers.  Think of it like a global
# variable that's properly scoped to the app instance.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manages the full lifecycle of all gateway subsystems.

    The pattern:
        [startup code]
        yield          ← server is live and accepting requests here
        [shutdown code]

    If anything in the startup block raises an exception, FastAPI will NOT
    start the server and the process will exit with an error.  This is the
    correct behaviour — a gateway with a broken security guard or no Redis
    connection should not silently start serving traffic.
    """

    # -----------------------------------------------------------------------
    # STARTUP
    # -----------------------------------------------------------------------

    # Load all settings from environment variables / .env file.
    # get_settings() is lru_cache-wrapped so it only reads the env once.
    settings = get_settings()

    log.info("Starting AI Gateway", version=settings.app_version, env=settings.environment)

    # --- Step 1: Redis connection pool ---
    #
    # We create ONE connection pool at startup and share it across:
    #   - The RateLimiter (sliding window state)
    #   - The SemanticCache (response payload storage with TTL)
    #
    # Using a pool (not a single connection) means multiple concurrent requests
    # can all talk to Redis simultaneously without waiting for each other.
    #
    # aioredis.from_url() is the async Redis client.  It does NOT actually
    # connect here — the first command will open the first connection lazily.
    redis_url = (
        f"redis://:{settings.redis.password}@{settings.redis.host}:{settings.redis.port}"
        if settings.redis.password
        else f"redis://{settings.redis.host}:{settings.redis.port}"
    )
    redis_client = aioredis.from_url(
        redis_url,
        db=settings.redis.db,
        max_connections=settings.redis.max_connections,
        decode_responses=True,  # return str instead of bytes
    )
    log.info("Redis connection pool created", host=settings.redis.host, port=settings.redis.port)

    # --- Step 2: Rate limiter ---
    #
    # The RateLimiter wraps the Redis client and adds the sliding-window logic.
    # We pass the RedisSettings slice (not the full Settings) — each module
    # only receives the config it actually needs.
    rate_limiter = RateLimiter(settings.redis)

    # --- Step 3: Security guard ---
    #
    # warm_up() loads the DistilBERT weights into memory.
    # This is the most expensive startup step (~2-5 seconds on CPU).
    # By doing it at startup, the FIRST real request does not pay this cost.
    security_guard = SecurityGuard(settings.security)
    await security_guard.warm_up()
    log.info("Security guard ready", ml_enabled=settings.security.ml_guard_enabled)

    # --- Step 4: Semantic cache ---
    #
    # warm_up() does two things:
    #   a) Loads the sentence-transformer embedding model into memory.
    #   b) Reads the FAISS index from disk (if it was persisted from a previous run).
    #      If no index file exists on disk yet, it creates a fresh empty one.
    #
    # We skip this entirely if caching is disabled in settings.
    semantic_cache = SemanticCache(settings.cache, redis_client)
    if settings.cache.enabled:
        await semantic_cache.warm_up()
        log.info("Semantic cache ready", index_size=semantic_cache._store.size)

    # --- Step 5: Contract registry ---
    #
    # load() scans the "contracts/" directory for JSON/YAML files and parses
    # each one into a ContractDefinition.  Non-fatal: files that fail to parse
    # are skipped with a warning.
    contract_registry = ContractRegistry("contracts/")
    contract_registry.load()
    log.info("Contract registry loaded", apps=contract_registry.list_apps())

    # --- Step 6: Provider router ---
    #
    # The router builds its alias registry from the settings and the built-in
    # model list.  No network calls here — it just populates an in-memory dict.
    provider_router = ProviderRouter(settings)

    # --- Step 7: Langfuse client ---
    #
    # Initialises the Langfuse SDK.  If LANGFUSE__ENABLED=false or the keys are
    # missing, this creates a no-op stub so the rest of the code can call
    # tracing methods without conditionals everywhere.
    langfuse_client = LangfuseClient(settings.langfuse)

    # --- Step 8: Metrics collector ---
    #
    # Simple in-memory counters and latency histograms.  No external dependency.
    metrics = MetricsCollector()

    # -----------------------------------------------------------------------
    # Store everything on app.state
    #
    # app.state is FastAPI's official mechanism for sharing objects between
    # the lifespan hook and request handlers.  It behaves like a plain object
    # — you can set any attribute on it.
    #
    # In request handlers we access these via: request.app.state.<name>
    # -----------------------------------------------------------------------
    app.state.settings = settings
    app.state.redis = redis_client
    app.state.rate_limiter = rate_limiter
    app.state.security_guard = security_guard
    app.state.semantic_cache = semantic_cache
    app.state.contract_registry = contract_registry
    app.state.provider_router = provider_router
    app.state.langfuse = langfuse_client
    app.state.metrics = metrics

    log.info("AI Gateway is ready to accept requests")

    # -----------------------------------------------------------------------
    # Yield — server is live here.
    # Everything below runs AFTER the server stops accepting new requests.
    # -----------------------------------------------------------------------
    yield

    # -----------------------------------------------------------------------
    # SHUTDOWN
    #
    # Order matters: stop accepting new work before closing shared resources.
    # -----------------------------------------------------------------------

    log.info("Shutting down AI Gateway...")

    # Flush any Langfuse events that are buffered in memory.
    # Without this, traces for the last few requests before shutdown are lost.
    langfuse_client.flush()

    # Persist the FAISS index to disk so the cache survives the restart.
    # The next startup will load this file instead of starting from scratch.
    if settings.cache.enabled:
        await semantic_cache.shutdown()
        log.info("FAISS index persisted to disk")

    # Close the Redis connection pool cleanly.
    # Prevents "connection reset" errors in the Redis server logs.
    await rate_limiter.close()
    await redis_client.aclose()

    log.info("Shutdown complete")


# ---------------------------------------------------------------------------
# App factory
#
# We use a factory function (create_app) instead of creating the app at module
# level directly because:
#   1. It's easier to test — tests can call create_app() to get a fresh instance
#      with dependency overrides injected.
#   2. It keeps the global namespace clean.
#   3. It makes the lifespan, middleware stack, and routers explicit and readable.
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Construct and return the fully configured FastAPI application instance.

    This function:
      1. Creates the FastAPI instance with metadata and the lifespan hook.
      2. Adds all middleware in the correct order (outermost to innermost).
      3. Returns the app so endpoints can be registered with @app decorators.

    Note on middleware ordering:
      FastAPI middleware wraps requests like an onion — the LAST middleware
      added is the OUTERMOST layer (first to see the request, last to see the
      response).  We add CORS last so it's outermost, which is the standard
      web convention.
    """

    # Create the FastAPI application.
    #
    # - title, description, version: shown in the auto-generated /docs UI
    # - lifespan: tells FastAPI to call our lifespan() hook at startup/shutdown
    application = FastAPI(
        title="AI Gateway",
        description=(
            "A self-hostable proxy layer for LLM providers. "
            "Drop-in OpenAI-compatible API with rate limiting, "
            "semantic caching, injection detection, and observability."
        ),
        version="0.1.0",
        docs_url="/docs",       # Swagger UI lives at /docs
        redoc_url="/redoc",     # ReDoc UI lives at /redoc
        lifespan=lifespan,      # wire up our startup/shutdown hook
    )

    # -----------------------------------------------------------------------
    # Middleware: Request ID injection
    #
    # Every request gets a unique UUID attached to it as the X-Request-ID header.
    # This ID threads through all log lines, Langfuse traces, and response
    # headers, making it trivial to find all log lines for a single request.
    #
    # If the client already sent an X-Request-ID header, we honour it (useful
    # for distributed tracing where the ID was assigned by an upstream service).
    # -----------------------------------------------------------------------
    @application.middleware("http")
    async def add_request_id(request: Request, call_next):
        # Use the client's ID if they sent one; otherwise generate a new UUID.
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Attach the ID to the request's state so downstream handlers can read it
        # without reparsing the header.
        request.state.request_id = request_id

        # Process the rest of the request pipeline
        response = await call_next(request)

        # Echo the request ID back in the response so clients can correlate
        # their requests with our logs.
        response.headers["X-Request-ID"] = request_id
        return response

    # -----------------------------------------------------------------------
    # Middleware: Request timing
    #
    # Measures the total wall-clock time from when NGINX handed the request to
    # uvicorn to when we finished writing the response.  Appears as the
    # X-Process-Time-Ms header on every response.
    #
    # This is NOT the same as the provider latency (which is measured inside
    # proxy_request).  This number includes Python overhead, middleware, etc.
    # -----------------------------------------------------------------------
    @application.middleware("http")
    async def add_process_time(request: Request, call_next):
        start = time.perf_counter()          # high-resolution timer (monotonic)
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000  # convert s → ms
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    # -----------------------------------------------------------------------
    # Middleware: CORS (Cross-Origin Resource Sharing)
    #
    # Browsers enforce the Same-Origin Policy: a web page at app.example.com
    # cannot call an API at api.example.com unless the API explicitly permits it
    # via CORS headers.
    #
    # For a developer gateway, we allow all origins ("*") by default.
    # In production, restrict allow_origins to your actual frontend domains.
    #
    # IMPORTANT: CORSMiddleware MUST be added AFTER other middleware so it is
    # the outermost layer and handles preflight OPTIONS requests before they
    # hit authentication or business logic.
    # -----------------------------------------------------------------------
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],               # TODO: restrict in production
        allow_credentials=False,           # must be False when allow_origins=["*"]
        allow_methods=["GET", "POST"],     # only the methods this API uses
        allow_headers=["*"],               # allow Authorization, Content-Type, etc.
    )

    return application


# ---------------------------------------------------------------------------
# Create the module-level app instance.
#
# uvicorn imports this when you run:
#   uvicorn gateway.main:app --host 0.0.0.0 --port 8000
#
# The lifespan hook fires automatically when uvicorn starts the ASGI server.
# ---------------------------------------------------------------------------
app: FastAPI = create_app()


# ---------------------------------------------------------------------------
# ENDPOINT: POST /v1/chat/completions
#
# The primary proxy endpoint. Mirrors the OpenAI Chat Completions API exactly
# so any OpenAI-compatible client can point at this gateway with zero code
# changes (just swap the base_url).
#
# Full pipeline — each stage is independently commented below.
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def proxy_request(request: Request) -> Response:
    """
    Core proxy handler. Runs the full gateway pipeline for every LLM request.

    The function intentionally does NO business logic itself — it delegates to
    the subsystems initialised in lifespan() and orchestrates their execution
    order.  Each subsystem can be tested, swapped, or disabled independently.
    """

    # -----------------------------------------------------------------------
    # Stage 0: Setup
    #
    # Capture the start time immediately so our latency measurement covers the
    # entire handler, including JSON parsing and auth.
    #
    # Pull subsystems from app.state.  This is how FastAPI shares objects that
    # were created at startup with per-request handlers.
    # -----------------------------------------------------------------------
    start_time = time.perf_counter()

    # Use the request ID set by the middleware (it's already in state).
    request_id: str = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Shorthand references to avoid typing "request.app.state." everywhere.
    settings: Settings = request.app.state.settings
    rate_limiter: RateLimiter = request.app.state.rate_limiter
    security_guard: SecurityGuard = request.app.state.security_guard
    semantic_cache: SemanticCache = request.app.state.semantic_cache
    contract_registry: ContractRegistry = request.app.state.contract_registry
    provider_router: ProviderRouter = request.app.state.provider_router
    langfuse: LangfuseClient = request.app.state.langfuse
    metrics: MetricsCollector = request.app.state.metrics

    # -----------------------------------------------------------------------
    # Stage 1: Parse the request body
    #
    # We read the raw JSON body manually (instead of using Pydantic models as
    # FastAPI parameters) because:
    #   a) We forward most of the body as-is to the provider.
    #   b) Different providers have slightly different schemas — we normalise
    #      them in the adapter layer, not here.
    #   c) It's easier to add custom gateway-specific fields to a raw dict.
    # -----------------------------------------------------------------------
    try:
        body: dict = await request.json()
    except Exception:
        # If the body is not valid JSON at all, reject immediately.
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "Request body must be valid JSON.", "type": "invalid_request_error"}},
        )

    # -----------------------------------------------------------------------
    # Stage 2: Authentication
    #
    # Validate the client's API key before doing ANY work.
    # We check auth AFTER parsing the body (so we can give a better error for
    # malformed JSON) but BEFORE the rate limiter (no point counting an
    # unauthenticated request against a legitimate key's quota).
    # -----------------------------------------------------------------------
    api_key = _authenticate(request, settings)

    # The app_id is used to look up behavioral contracts.  Clients can send a
    # custom X-App-ID header to identify themselves; otherwise we use the first
    # 8 characters of their API key as a stable identifier.
    app_id = request.headers.get("X-App-ID", api_key[:8])

    # -----------------------------------------------------------------------
    # Stage 3: Rate limiting
    #
    # We enforce a sliding-window rate limit per API key.
    #
    # WindowConfig defines the rule: max N requests per window_seconds seconds.
    # The key prefix "rl:apikey:" namespaces rate limit keys in Redis so they
    # don't collide with cache keys.
    #
    # If the limit is exceeded, RateLimiter.check() raises RateLimitExceeded,
    # which we catch and convert to an HTTP 429 response with a Retry-After
    # header — this is the standard HTTP convention for rate limiting.
    # -----------------------------------------------------------------------
    window_config = WindowConfig(
        max_requests=100,   # 100 requests...
        window_seconds=settings.redis.rate_limit_window_seconds,  # ...per minute
        key_prefix="rl:apikey",
    )

    try:
        await rate_limiter.check(key=api_key, config=window_config)
    except RateLimitExceeded as exc:
        metrics.record_rate_limited()
        log.warning("Rate limit exceeded", api_key=api_key[:8], retry_after=exc.retry_after)

        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": f"Rate limit exceeded. Retry after {exc.retry_after:.1f} seconds.",
                    "type": "rate_limit_error",
                    "retry_after": exc.retry_after,
                }
            },
            headers={
                # Retry-After tells the client exactly when to try again.
                # It's specified by RFC 7231 and respected by most HTTP clients.
                "Retry-After": str(int(exc.retry_after)),
                "X-RateLimit-Limit": str(exc.limit),
                "X-RateLimit-Window": str(exc.window),
            },
        )

    # -----------------------------------------------------------------------
    # Stage 4: Prompt injection detection
    #
    # The SecurityGuard runs a two-tier scan:
    #   Tier 1 (PatternGuard): regex patterns, ~0.5 ms, catches obvious attacks.
    #   Tier 2 (MLGuard): DistilBERT classifier, ~30-100 ms on CPU, catches
    #                     sophisticated attacks that evade regex.
    #
    # _extract_prompt_text() flattens the messages array into a single string
    # that both tiers can operate on.
    #
    # If an injection is detected, we return HTTP 400 (Bad Request) — we don't
    # want to give the attacker a 403 (Forbidden) which implies the content was
    # understood but rejected; 400 implies we couldn't process the request.
    # -----------------------------------------------------------------------
    prompt_text = _extract_prompt_text(body)

    guard_result = await security_guard.scan(prompt_text)

    if guard_result.blocked:
        # Record which tier caught it for the metrics dashboard.
        metrics.record_injection_blocked(tier=guard_result.tier_triggered or 1)
        log.warning(
            "Injection detected",
            tier=guard_result.tier_triggered,
            reason=guard_result.reason,
            api_key=api_key[:8],
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": guard_result.reason,
                    "type": "prompt_injection_detected",
                    "tier": guard_result.tier_triggered,
                }
            },
        )

    # -----------------------------------------------------------------------
    # Stage 5: Semantic cache lookup
    #
    # Before calling any provider (which costs money and takes 200-2000 ms),
    # check if we have a semantically equivalent response cached.
    #
    # "Semantically equivalent" means the incoming prompt is within a cosine
    # similarity threshold (e.g. 0.92) of a previously-seen prompt.  This
    # catches paraphrased questions like:
    #   "What is the capital of France?" ≈ "Which city is France's capital?"
    #
    # On a HIT: return the cached response immediately with X-Cache: HIT.
    # On a MISS: proceed to provider (we'll store the response later at Stage 9).
    # -----------------------------------------------------------------------
    if settings.cache.enabled:
        cache_hit = await semantic_cache.get(prompt_text)

        if cache_hit is not None:
            # Cache hit — we can skip the provider entirely.
            metrics.record_cache_hit()
            log.info(
                "Cache hit",
                similarity=round(cache_hit.similarity, 4),
                cache_key=cache_hit.cache_key,
            )
            return JSONResponse(
                content=cache_hit.payload,
                headers={
                    "X-Cache": "HIT",
                    "X-Cache-Similarity": f"{cache_hit.similarity:.4f}",
                    "X-Cache-Key": cache_hit.cache_key,
                },
            )

        # Cache miss — record it and continue to provider.
        metrics.record_cache_miss()

    # -----------------------------------------------------------------------
    # Stage 6: Provider routing
    #
    # The ProviderRouter examines the "model" field in the request body and
    # resolves it to:
    #   - which upstream provider to use (OpenAI, Anthropic, or Ollama)
    #   - which canonical model ID to send to that provider
    #   - which adapter class to instantiate
    #
    # This is where "gpt-4o" → (provider="openai", model="gpt-4o-2024-11-20")
    # or "claude-3-5-sonnet" → (provider="anthropic", model="claude-3-5-sonnet-20241022")
    #
    # The routing decision can be influenced by:
    #   - The X-Provider header (explicit override)
    #   - The active routing policy (COST or LATENCY)
    # -----------------------------------------------------------------------
    try:
        routing_decision = provider_router.resolve(body)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": str(exc), "type": "invalid_request_error"}},
        )

    log.info(
        "Routing decision",
        provider=routing_decision.provider_name,
        model=routing_decision.canonical_model,
        alias=routing_decision.alias_used,
    )

    # -----------------------------------------------------------------------
    # Stage 7: Call the upstream provider
    #
    # We instantiate the adapter class from the routing decision.  Each adapter
    # knows how to:
    #   - Translate our internal OpenAI-schema request to the provider's format
    #   - Make the HTTP request (with retry logic)
    #   - Translate the response back to our internal format
    #
    # We measure provider latency separately from total request latency so we
    # can distinguish "slow provider" from "slow gateway overhead" in metrics.
    # -----------------------------------------------------------------------
    adapter = routing_decision.adapter_class(settings.providers)
    provider_start = time.perf_counter()

    try:
        response_body = await adapter.complete(body)
    except Exception as exc:
        # Any exception from the provider (network error, auth failure, 5xx, etc.)
        # becomes a 502 Bad Gateway — the standard code for "upstream failed".
        metrics.record_provider_error(routing_decision.provider_name)
        log.error(
            "Provider error",
            provider=routing_decision.provider_name,
            error=str(exc),
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": f"Upstream provider error: {exc}",
                    "type": "provider_error",
                    "provider": routing_decision.provider_name,
                }
            },
        )

    provider_latency_ms = (time.perf_counter() - provider_start) * 1000

    # -----------------------------------------------------------------------
    # Stage 8: Behavioral contract evaluation
    #
    # Behavioral contracts are rules the application owner defines to ensure
    # the LLM's responses meet their requirements.  Examples:
    #   - "Never mention competitor brand names"
    #   - "All responses must be in JSON format"
    #   - "Sentiment must be >= 0.3 (not hostile)"
    #
    # We evaluate all contracts registered for this app_id.
    # If any BLOCK-action contract is violated, we return HTTP 422.
    # WARN-action violations are logged but the response is still returned.
    # -----------------------------------------------------------------------
    contract_report = await contract_registry.evaluate(
        app_id=app_id,
        request=body,
        response=response_body,
    )

    if contract_report.blocked:
        # A BLOCK-action contract was violated.
        # We do NOT return the provider's response — it violated the rules.
        metrics.record_contract_violation()
        violated_ids = [v.contract_id for v in contract_report.violations]
        log.warning("Contract violation — response blocked", contracts=violated_ids, app_id=app_id)

        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": "Response blocked by behavioral contract.",
                    "type": "contract_violation",
                    "violated_contracts": violated_ids,
                }
            },
        )

    # -----------------------------------------------------------------------
    # Stage 9: Store in semantic cache
    #
    # Now that we have a valid, contract-approved response, store it.
    # Future requests with semantically similar prompts will get this cached
    # response instead of hitting the provider.
    #
    # We store AFTER contract evaluation to ensure we never cache a response
    # that violated a contract (in case a violation slips through on one
    # request and then gets served to future callers from cache).
    # -----------------------------------------------------------------------
    if settings.cache.enabled:
        await semantic_cache.set(prompt_text, response_body)

    # -----------------------------------------------------------------------
    # Stage 10: Emit Langfuse trace
    #
    # Record the full request in Langfuse for observability.
    # We do this AFTER building the response so we can include the final output.
    #
    # extract_usage() pulls token counts from the normalised response — these
    # are used by Langfuse to calculate cost estimates in the dashboard.
    # -----------------------------------------------------------------------
    usage = adapter.extract_usage(response_body)

    async with langfuse.create_trace(
        request_id=request_id,
        app_id=app_id,
        metadata={
            "provider": routing_decision.provider_name,
            "model": routing_decision.canonical_model,
            "alias": routing_decision.alias_used,
            "cache": "MISS",
            "security_tier": guard_result.tier_triggered,
            "contract_violations": len(contract_report.violations),
        },
    ) as trace:
        # Record the LLM generation with token usage for cost tracking.
        trace.record_generation(
            model=routing_decision.canonical_model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        trace.set_output(response_body)

    # -----------------------------------------------------------------------
    # Stage 11: Update metrics and routing stats
    #
    # MetricsCollector accumulates counters and latency samples in memory.
    # These are exposed at GET /metrics for dashboards.
    #
    # We also update the router's rolling latency average so the LATENCY
    # routing policy has fresh data for the next request.
    # -----------------------------------------------------------------------
    total_latency_ms = (time.perf_counter() - start_time) * 1000

    metrics.record_request(
        provider=routing_decision.provider_name,
        model=routing_decision.canonical_model,
        latency_ms=total_latency_ms,
    )

    provider_router.update_latency(
        provider_name=routing_decision.provider_name,
        alias=routing_decision.alias_used,
        latency_ms=provider_latency_ms,
    )

    log.info(
        "Request complete",
        provider=routing_decision.provider_name,
        model=routing_decision.canonical_model,
        total_ms=round(total_latency_ms, 1),
        provider_ms=round(provider_latency_ms, 1),
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )

    # -----------------------------------------------------------------------
    # Return the final response.
    #
    # We add gateway-specific headers so clients can see exactly what happened:
    #   X-Cache: MISS  — this request hit the provider (not cache)
    #   X-Provider     — which upstream provider was used
    #   X-Model        — the canonical model ID the provider used
    #   X-Latency-Ms   — total gateway latency
    #   X-Provider-Latency-Ms — just the provider round-trip time
    # -----------------------------------------------------------------------
    return JSONResponse(
        content=response_body,
        headers={
            "X-Cache": "MISS",
            "X-Provider": routing_decision.provider_name,
            "X-Model": routing_decision.canonical_model,
            "X-Latency-Ms": f"{total_latency_ms:.1f}",
            "X-Provider-Latency-Ms": f"{provider_latency_ms:.1f}",
        },
    )


# ---------------------------------------------------------------------------
# ENDPOINT: GET /health
#
# Used by:
#   - Docker healthcheck (HEALTHCHECK CMD curl -f http://localhost:8000/health)
#   - Kubernetes liveness and readiness probes
#   - NGINX upstream health check (nginx_http_check module)
#   - Monitoring systems (Uptime Robot, Datadog synthetics, etc.)
#
# Returns 200 when the gateway is healthy, 503 when any critical subsystem
# is unavailable.
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check(request: Request) -> JSONResponse:
    """
    Liveness and readiness probe.

    Checks each subsystem and reports its status.  Returns HTTP 200 when
    everything is healthy, HTTP 503 (Service Unavailable) if a critical
    subsystem is down — which tells the load balancer to stop routing
    traffic to this instance.
    """
    status = {
        "status": "ok",     # top-level status, overwritten to "degraded" if anything fails
        "version": "0.1.0",
        "subsystems": {},
    }

    all_healthy = True

    # --- Check Redis ---
    # We send a PING command.  Redis replies "PONG" if it's reachable.
    # This also validates that our connection pool is functional.
    try:
        redis_client = request.app.state.redis
        await redis_client.ping()
        status["subsystems"]["redis"] = "ok"
    except Exception as exc:
        status["subsystems"]["redis"] = f"error: {exc}"
        all_healthy = False  # Redis is critical — rate limiting will fail

    # --- Check FAISS index ---
    # We just check that the index object exists in memory and has been built.
    # We don't run a search query here (too slow for a health probe).
    try:
        cache = request.app.state.semantic_cache
        index_size = cache._store.size
        status["subsystems"]["faiss_index"] = f"ok ({index_size} vectors)"
    except Exception as exc:
        status["subsystems"]["faiss_index"] = f"error: {exc}"
        # Cache failure is not critical — the gateway degrades gracefully.

    # --- Check ML model ---
    # We verify the MLGuard's model attribute is loaded in memory.
    # We don't run an inference pass (that would add 30-100 ms to every probe).
    try:
        guard = request.app.state.security_guard
        ml_loaded = guard._ml_guard._model is not None
        status["subsystems"]["ml_model"] = "ok (loaded)" if ml_loaded else "not loaded"
    except Exception as exc:
        status["subsystems"]["ml_model"] = f"error: {exc}"
        # ML failure is not critical if pattern guard is still running.

    # If any critical subsystem is down, return 503 so orchestrators know
    # to not route traffic to this instance.
    http_status = 200 if all_healthy else 503
    if not all_healthy:
        status["status"] = "degraded"

    return JSONResponse(content=status, status_code=http_status)


# ---------------------------------------------------------------------------
# ENDPOINT: GET /metrics
#
# Exposes in-memory metrics aggregated by MetricsCollector.
# Useful for:
#   - A simple Grafana dashboard (connect directly to this endpoint)
#   - Prometheus scraping (add a /metrics/prometheus endpoint later)
#   - A quick curl during debugging: curl http://localhost:8000/metrics | jq
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def get_metrics(request: Request) -> JSONResponse:
    """
    Return a snapshot of all in-memory gateway metrics.

    The snapshot is a point-in-time copy — concurrent requests that complete
    after this endpoint is called are not included.  This is acceptable for
    a dashboard polling endpoint (it re-fetches every N seconds).

    For high-cardinality production metrics, integrate Prometheus +
    prometheus-fastapi-instrumentator instead of this endpoint.
    """
    metrics: MetricsCollector = request.app.state.metrics

    # snapshot() returns a JSON-safe dict — no numpy types, no non-serialisable
    # objects.  See MetricsCollector.snapshot() for the full schema.
    return JSONResponse(content=metrics.snapshot())


# ---------------------------------------------------------------------------
# ENDPOINT: GET /v1/models
#
# OpenAI-compatible endpoint that lists all models the gateway can route to.
# Useful for clients that call this to discover available models before
# sending requests (e.g. the OpenAI Python SDK does this automatically).
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    """
    List all registered model aliases in OpenAI's /v1/models format.

    Response shape matches the OpenAI API so OpenAI-compatible clients
    can use their normal model listing calls without modification.
    """
    router: ProviderRouter = request.app.state.provider_router
    models = router.list_models()

    return JSONResponse(
        content={
            "object": "list",
            "data": models,
        }
    )
