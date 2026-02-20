# AI Gateway

> A self-hostable, drop-in infrastructure layer that sits in front of LLM providers.

---

## Overview

## Architecture

## Features

- **Provider Routing** — Unified API across OpenAI, Anthropic, and local Ollama models with cost/latency-aware routing policies
- **Rate Limiting** — Sliding-window rate limiting backed by Redis; consistent across multiple gateway replicas
- **Prompt Injection Detection** — Two-tier pipeline: regex pattern matching (< 1 ms) + DistilBERT ML classifier
- **Semantic Caching** — FAISS-backed approximate nearest-neighbour cache; returns cached responses for paraphrased prompts
- **Behavioral Contracts** — Declarative rules (keyword blocks, regex, JSON Schema, sentiment) evaluated against every response
- **Observability** — Langfuse distributed tracing with per-request spans for every pipeline stage
- **Single-command Deploy** — `docker compose up` starts the full stack

---

## Getting Started

### Prerequisites

### Installation

### Configuration

### Running with Docker Compose

### Running Locally (Development)

---

## API Reference

### `POST /v1/chat/completions`

### `GET /health`

### `GET /metrics`

### `GET /v1/models`

---

## Provider Routing

### Supported Providers

### Model Alias Registry

### Routing Policies

### Adding a New Provider

---

## Rate Limiting

### Algorithm

### Configuration

### Per-Key and Per-Model Limits

---

## Prompt Injection Detection

### Tier 1 — Pattern Guard

### Tier 2 — ML Guard (DistilBERT)

### Training Your Own Classifier

### Tuning the Confidence Threshold

---

## Semantic Cache

### How It Works

### Configuration

### Similarity Threshold Tuning

### Cache Persistence

---

## Behavioral Contracts

### Contract Types

### Writing a Contract Definition

### Registering Contracts

### Contract Actions (BLOCK / WARN / LOG)

---

## Observability

### Langfuse Tracing

### Metrics Endpoint

### Log Format

---

## Deployment

### Docker Compose (recommended)

### Environment Variables

### TLS / HTTPS

### Scaling Horizontally

### Using a GPU for the ML Guard

---

## Development

### Project Structure

### Running Tests

### Linting and Formatting

### Generating Test Data

### Benchmarking the Security Pipeline

---

## Contributing

## License

## Acknowledgements
