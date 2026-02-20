#!/usr/bin/env python3
"""
Download all ML models required by Covenant before first run.

Usage:
    python scripts/download_models.py                 # core models only
    DOWNLOAD_ALL_MODELS=1 python scripts/download_models.py  # + benchmark models

Models downloaded:
  - sentence-transformers/all-MiniLM-L6-v2       (embedder, ~80 MB)
  - ProtectAI/deberta-v3-base-prompt-injection-v2 (ML guard, ~700 MB)

Optional (for benchmarking — set DOWNLOAD_ALL_MODELS=1):
  - meta-llama/Llama-Prompt-Guard-2-86M          (requires HF login + Llama license)
  - meta-llama/Llama-Prompt-Guard-2-22M          (requires HF login + Llama license)

NOTE: The Meta Prompt Guard models require:
  1. A HuggingFace account (https://huggingface.co/join)
  2. Accepting the Llama license at the model page
  3. Running `huggingface-cli login` before this script
"""

import os
import sys


MODELS = {
    "embedder": "sentence-transformers/all-MiniLM-L6-v2",
    "ml_guard_default": "ProtectAI/deberta-v3-base-prompt-injection-v2",
    "ml_guard_meta_86m": "meta-llama/Llama-Prompt-Guard-2-86M",
    "ml_guard_meta_22m": "meta-llama/Llama-Prompt-Guard-2-22M",
}


def download_embedder(model_id: str) -> None:
    from sentence_transformers import SentenceTransformer
    print(f"Downloading embedder: {model_id}")
    SentenceTransformer(model_id)
    print(f"  done — {model_id}")


def download_classifier(model_id: str) -> None:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"Downloading classifier: {model_id}")
    AutoTokenizer.from_pretrained(model_id)
    AutoModelForSequenceClassification.from_pretrained(model_id)
    print(f"  done — {model_id}")


def main() -> int:
    print("Covenant — Model Downloader")
    print("=" * 50)

    try:
        download_embedder(MODELS["embedder"])
    except Exception as exc:
        print(f"  FAILED: {exc}", file=sys.stderr)
        return 1

    try:
        download_classifier(MODELS["ml_guard_default"])
    except Exception as exc:
        print(f"  FAILED: {exc}", file=sys.stderr)
        return 1

    if os.getenv("DOWNLOAD_ALL_MODELS"):
        print("\nDownloading benchmark models (Meta Prompt Guard)...")
        print("NOTE: These require `huggingface-cli login` and Llama license acceptance.")

        for key in ("ml_guard_meta_86m", "ml_guard_meta_22m"):
            try:
                download_classifier(MODELS[key])
            except Exception as exc:
                print(f"  SKIPPED {MODELS[key]}: {exc}")
                print("  (Have you accepted the Llama license and run `huggingface-cli login`?)")

    print("\nAll core models downloaded. Covenant is ready to start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
