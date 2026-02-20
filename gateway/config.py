"""
gateway/config.py

Centralised settings management using Pydantic-Settings.

All runtime configuration is sourced from environment variables (or a .env
file during local development).  A single `Settings` instance is constructed
once and injected wherever needed via FastAPI's dependency-injection system or
imported directly.

Responsibilities:
  - Declare every environment variable the gateway needs as typed fields with
    sensible defaults and validation rules.
  - Group settings into logical nested models (RedisSettings, LangfuseSettings,
    ProviderSettings, SecuritySettings, CacheSettings) so that each module can
    receive only the slice it needs.
  - Expose a cached `get_settings()` function that can be used as a FastAPI
    dependency.

Key classes / functions:
  - RedisSettings        — host, port, password, TTL, max connections
  - LangfuseSettings     — public key, secret key, host, flush interval
  - ProviderSettings     — API keys and base URLs for OpenAI, Anthropic, Ollama
  - SecuritySettings     — ML model path, confidence threshold, pattern file path
  - CacheSettings        — FAISS index path, similarity threshold, embedding model name
  - Settings             — root model that composes all sub-settings
  - get_settings()       — lru_cache-wrapped factory, safe for use as a FastAPI dep
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-settings models
# ---------------------------------------------------------------------------

class RedisSettings(BaseSettings):
    """
    Connection parameters and behavioural tuning for the Redis backend used by
    the distributed rate limiter.
    """
    host: str = Field("redis", description="Redis hostname")
    port: int = Field(6379, description="Redis port")
    password: Optional[str] = Field(None, description="Redis password (if auth enabled)")
    db: int = Field(0, description="Redis database index")
    max_connections: int = Field(20, description="Max connections in the pool")
    rate_limit_window_seconds: int = Field(60, description="Sliding window size in seconds")

    # TODO: add SSL/TLS fields when needed


class LangfuseSettings(BaseSettings):
    """
    Credentials and transport options for the Langfuse observability backend.
    """
    public_key: str = Field("", description="Langfuse public key")
    secret_key: str = Field("", description="Langfuse secret key")
    host: str = Field("https://cloud.langfuse.com", description="Langfuse server URL")
    flush_interval: float = Field(0.5, description="Seconds between automatic batch flushes")
    enabled: bool = Field(True, description="Set to false to disable tracing entirely")


class ProviderSettings(BaseSettings):
    """
    API credentials and base URLs for each supported upstream LLM provider.
    """
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_base_url: str = Field("https://api.openai.com/v1", description="OpenAI compatible base URL")

    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    anthropic_base_url: str = Field("https://api.anthropic.com", description="Anthropic API base URL")

    ollama_base_url: str = Field("http://ollama:11434", description="Ollama server base URL")

    default_provider: str = Field("openai", description="Fallback provider when none specified")
    request_timeout_seconds: int = Field(60, description="Per-request timeout to upstream provider")


class SecuritySettings(BaseSettings):
    """
    Parameters for both tiers of the prompt injection detection pipeline.
    """
    # Tier 1 — pattern guard
    pattern_file_path: str = Field(
        "gateway/security/patterns.json",
        description="Path to JSON file containing injection regex patterns",
    )
    pattern_guard_enabled: bool = Field(True, description="Enable regex pattern scanning")

    # Tier 2 — ML guard
    ml_model_path: str = Field(
        "models/injection_classifier",
        description="Local path to the fine-tuned DistilBERT classifier",
    )
    ml_guard_enabled: bool = Field(True, description="Enable ML-based injection detection")
    ml_confidence_threshold: float = Field(
        0.85,
        ge=0.0,
        le=1.0,
        description="Minimum model confidence to block a request",
    )

    @field_validator("ml_confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Ensure the threshold is a valid probability."""
        # TODO: implement
        return v


class CacheSettings(BaseSettings):
    """
    Parameters for the FAISS-backed semantic response cache.
    """
    enabled: bool = Field(True, description="Enable semantic caching")
    embedding_model_name: str = Field(
        "all-MiniLM-L6-v2",
        description="Sentence-transformers model used for prompt embeddings",
    )
    faiss_index_path: str = Field(
        "cache/faiss.index",
        description="Filesystem path where the FAISS index is persisted",
    )
    similarity_threshold: float = Field(
        0.92,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score above which a cache hit is declared",
    )
    max_cache_entries: int = Field(10_000, description="Maximum vectors stored in the FAISS index")
    cache_ttl_seconds: int = Field(3600, description="Seconds before a cache entry expires (Redis TTL)")


class ContractSettings(BaseSettings):
    """
    Parameters for the behavioral contracts layer.
    """
    contracts_dir: str = Field("contracts/", description="Directory containing contract JSON files")
    drift_enabled: bool = Field(True, description="Enable drift detection via Redis time series")
    drift_recent_window_hours: float = Field(
        24.0,
        description="Recent window for drift comparison (hours)",
    )
    drift_baseline_window_hours: float = Field(
        168.0,
        description="Baseline window for drift comparison (hours, default 7 days)",
    )
    drift_alert_threshold: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description="Relative compliance drop that triggers a drift alert (0.10 = 10%)",
    )
    drift_min_samples: int = Field(
        10,
        ge=1,
        description="Minimum samples required before drift detection fires",
    )
    drift_retention_days: int = Field(
        30,
        ge=1,
        description="How many days of compliance data to retain in Redis",
    )
    drift_check_interval_seconds: int = Field(
        300,
        description="How often to run drift checks in the background (seconds)",
    )


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Root settings object.  Composed of all sub-settings groups plus
    top-level gateway configuration values.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Gateway identity
    app_name: str = Field("AI Gateway", description="Human-readable service name")
    app_version: str = Field("0.1.0", description="Semantic version string")
    environment: str = Field("development", description="One of: development, staging, production")
    log_level: str = Field("INFO", description="Python logging level")
    gateway_api_key: str = Field("", description="Master API key required on all inbound requests")

    # Sub-settings (populated from nested env vars via env_nested_delimiter)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    contracts: ContractSettings = Field(default_factory=ContractSettings)


# ---------------------------------------------------------------------------
# Dependency / singleton accessor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance, constructed once from environment
    variables.  Safe to use as a FastAPI dependency:

        @app.get("/")
        def endpoint(settings: Settings = Depends(get_settings)):
            ...

    Why lru_cache(maxsize=1)?
      Pydantic-Settings reads and validates ALL environment variables every time
      you call Settings().  That involves disk I/O (.env file), environment
      lookups, and Pydantic validation.  By caching the result we pay this cost
      exactly once at startup, and every subsequent call returns the same
      already-constructed Settings object in microseconds.

      maxsize=1 means "cache exactly one call" — since there are no arguments,
      the first call's result is returned for every subsequent call.

    In tests, clear the cache between test cases with:
        get_settings.cache_clear()
    """
    # Settings() triggers Pydantic-Settings to:
    #   1. Read the .env file (if it exists)
    #   2. Read matching environment variables (env vars override .env file)
    #   3. Validate every field (wrong type → ValidationError with a clear message)
    #   4. Return the fully-populated Settings instance
    return Settings()
