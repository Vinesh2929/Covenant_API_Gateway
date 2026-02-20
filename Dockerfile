# =============================================================================
# AI Gateway — Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder) — install Python dependencies into a virtual environment
#   Stage 2 (runtime) — copy only the venv and application source; no build tools
#
# The resulting image is lean: no pip, no gcc, no build headers in production.
#
# Build:
#   docker build -t ai-gateway:latest .
#
# Run locally (without Compose):
#   docker run --env-file .env -p 8000:8000 ai-gateway:latest
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder — install dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System build deps for packages that compile C extensions (faiss, numpy, etc.)
# TODO: add: apt-get install -y --no-install-recommends build-essential libgomp1

# Copy only requirements first to leverage Docker layer caching.
# The venv is rebuilt only when requirements.txt changes, not on every code edit.
COPY requirements.txt .

# Create a venv and install into it (keeps the runtime stage clean)
# TODO: python -m venv /opt/venv
# TODO: /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2: runtime — lean production image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user for security (do not run as root in production)
# TODO: addgroup --system gateway && adduser --system --ingroup gateway gateway

WORKDIR /app

# Copy the virtual environment from the builder stage
# TODO: COPY --from=builder /opt/venv /opt/venv
# TODO: ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY gateway/ ./gateway/
COPY scripts/  ./scripts/

# Directories that are mounted as Docker volumes at runtime
# (models, cache, contracts, patterns) — create empty placeholders
RUN mkdir -p models cache contracts gateway/security

# TODO: chown -R gateway:gateway /app

# Switch to non-root user
# TODO: USER gateway

# Expose the port uvicorn listens on (NGINX proxies to this)
EXPOSE 8000

# Healthcheck — Docker / orchestrator uses this to detect unhealthy containers
# TODO: HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#         CMD curl -f http://localhost:8000/health || exit 1

# Default command — overridden in docker-compose.yml for staging/production
# TODO: CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
