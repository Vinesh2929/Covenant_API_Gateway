# =============================================================================
# AI Gateway — Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder) — install Python dependencies into a virtual environment
#   Stage 2 (runtime) — copy only the venv and application source; no build tools
#
# WHY MULTI-STAGE?
#   Building packages like faiss-cpu, torch, and sentence-transformers requires
#   gcc, g++, and various header files.  We don't want those in the final image
#   (increases attack surface and image size by ~400 MB).  Multi-stage lets us
#   compile in a fat builder image, then copy only the finished artifacts.
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
# python:3.11-slim is the official slim variant (~50 MB) — it has the Python
# runtime but no dev tools.  We add build tools below for C-extension packages.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build dependencies needed to compile C/C++ extension packages:
#   build-essential — gcc, g++, make
#   libgomp1        — OpenMP threading (required by FAISS for parallel search)
#   git             — some pip packages fetch source via git
# --no-install-recommends keeps the layer small by skipping optional packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching.
# The venv is rebuilt ONLY when requirements.txt changes, not on every code edit.
# This is the most important caching trick in Dockerfile best practices.
COPY requirements.txt .

# Create a virtual environment inside /opt/venv.
# Installing into a venv (rather than system Python) makes it trivial to
# copy just the venv to the runtime stage — no stray files left in /usr/local.
RUN python -m venv /opt/venv

# Install all Python dependencies into the venv.
# --no-cache-dir: don't cache downloaded wheels (saves ~100 MB in the layer).
RUN /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2: runtime — lean production image
# ---------------------------------------------------------------------------
# Start fresh from the same slim base — no build tools, no compiler cache.
FROM python:3.11-slim AS runtime

# Install only the minimal runtime shared libraries that the compiled packages
# need at runtime (not at build time):
#   libgomp1 — OpenMP runtime for FAISS parallel search
#   curl     — used by the HEALTHCHECK command below
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security.
# Running as root inside a container is a bad practice — if the process is
# compromised, the attacker has root access to the container's filesystem.
RUN addgroup --system gateway \
    && adduser --system --ingroup gateway --no-create-home gateway

WORKDIR /app

# Copy the virtual environment from the builder stage.
# This brings in all Python packages without pip or build tools.
COPY --from=builder /opt/venv /opt/venv

# Make the venv's Python and binaries the default by prepending to PATH.
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source.
# We copy gateway/ and scripts/ but NOT requirements.txt, tests/, or .git/
# to keep the image small and avoid leaking dev artifacts.
COPY gateway/ ./gateway/
COPY scripts/  ./scripts/

# Create directories for data that is mounted as Docker volumes at runtime.
# These directories are created as placeholders so the mount points exist.
# The actual data (ML models, FAISS index, contract files) lives on the host
# and is injected via `volumes:` in docker-compose.yml.
#   models/    — DistilBERT / DeBERTa checkpoint directory
#   cache/     — FAISS index persistence (faiss.index + faiss.index.meta)
#   contracts/ — application behavioral contract JSON files
RUN mkdir -p models cache contracts

# Transfer ownership to the non-root user AFTER creating all directories.
RUN chown -R gateway:gateway /app

# Drop root privileges — all subsequent commands (and the process) run as
# the non-privileged "gateway" user.
USER gateway

# Expose the port uvicorn listens on.
# NGINX (defined in docker-compose.yml) reverse-proxies to this port.
EXPOSE 8000

# Healthcheck — Docker daemon and Kubernetes use this to decide whether the
# container is healthy and ready to receive traffic.
#   --interval=30s:    check every 30 seconds
#   --timeout=10s:     fail the check if no response within 10 seconds
#   --start-period=60s: give the app 60 seconds to start before checks begin
#                       (ML model warm-up can take 30-60 seconds on first run)
#   --retries=3:       mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command — starts uvicorn with a single worker process.
# In production, the docker-compose.yml `command:` override sets the worker
# count to match CPU cores (e.g. --workers 4).
# We use a single worker here so the Dockerfile is safe to run standalone.
CMD ["uvicorn", "gateway.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-config", "/dev/null"]
