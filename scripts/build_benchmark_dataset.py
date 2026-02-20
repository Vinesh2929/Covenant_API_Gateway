#!/usr/bin/env python3
"""
scripts/build_benchmark_dataset.py

Build the labeled JSONL dataset used by benchmark_security.py.

Sources:
  Injections: deepset/prompt-injections (HuggingFace) — 662 adversarial examples,
              publicly available, no license restrictions.  Widely cited in academic
              prompt-injection papers, making the benchmark reproducible.

  Clean:      tatsu-lab/alpaca — 52K real user instructions, no injections.
              Sampled uniformly so the clean set covers diverse topics.

Output format (one JSON object per line):
  {"text": "...", "label": 0}   ← clean
  {"text": "...", "label": 1}   ← injection

Target split: 50/50 so precision/recall are directly comparable without reweighting.

Usage:
    pip install datasets
    python scripts/build_benchmark_dataset.py
    python scripts/build_benchmark_dataset.py --n-each 750 --seed 99
    python scripts/build_benchmark_dataset.py --output data/my_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def build(n_each: int = 500, seed: int = 42, output: str = "data/benchmark_dataset.jsonl") -> None:
    """
    Pull injection + clean samples, balance, shuffle, and write to JSONL.

    Args:
        n_each: Number of samples for EACH class. Total = 2 * n_each.
        seed:   Random seed for reproducibility.
        output: Output JSONL path.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed — run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 1. Injection samples — deepset/prompt-injections
    # ------------------------------------------------------------------
    print("Loading deepset/prompt-injections …")
    pi_ds = load_dataset("deepset/prompt-injections", split="train")

    # Dataset has columns: text, label (0=safe, 1=injection)
    # Keep only the injection class (label == 1)
    injection_texts = [row["text"] for row in pi_ds if row["label"] == 1]
    print(f"  Found {len(injection_texts)} injection samples")

    if len(injection_texts) < n_each:
        print(
            f"  Warning: only {len(injection_texts)} injections available "
            f"(requested {n_each}). Using all of them and adjusting n_each.",
            file=sys.stderr,
        )
        n_each = min(n_each, len(injection_texts))

    rng.shuffle(injection_texts)
    injection_samples = [
        {"text": t, "label": 1} for t in injection_texts[:n_each]
    ]

    # ------------------------------------------------------------------
    # 2. Clean samples — tatsu-lab/alpaca
    # ------------------------------------------------------------------
    print("Loading tatsu-lab/alpaca for clean prompts …")
    alpaca_ds = load_dataset("tatsu-lab/alpaca", split="train")

    # Use the 'instruction' field — these are real user-generated instructions,
    # not injections.  Filter out very short or very long texts.
    clean_texts = [
        row["instruction"]
        for row in alpaca_ds
        if 20 <= len(row["instruction"]) <= 512
    ]
    print(f"  Found {len(clean_texts)} candidate clean samples")

    if len(clean_texts) < n_each:
        print(f"  Warning: only {len(clean_texts)} clean samples available.", file=sys.stderr)
        n_each = min(n_each, len(clean_texts))

    rng.shuffle(clean_texts)
    clean_samples = [
        {"text": t, "label": 0} for t in clean_texts[:n_each]
    ]

    # ------------------------------------------------------------------
    # 3. Balance, shuffle, write
    # ------------------------------------------------------------------
    # Re-balance if one class has fewer samples than the other
    actual_n = min(len(injection_samples), len(clean_samples))
    injection_samples = injection_samples[:actual_n]
    clean_samples = clean_samples[:actual_n]

    all_samples = injection_samples + clean_samples
    rng.shuffle(all_samples)

    n_inj = sum(1 for s in all_samples if s["label"] == 1)
    n_clean = sum(1 for s in all_samples if s["label"] == 0)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nDataset written to {output_path}")
    print(f"  Total samples : {len(all_samples)}")
    print(f"  Injections (1): {n_inj}")
    print(f"  Clean      (0): {n_clean}")
    print(f"  Seed            : {seed}")
    print(
        "\nPaste these provenance details into CONTEXT.md when recording results:\n"
        f"  Injection source : deepset/prompt-injections (HuggingFace)\n"
        f"  Clean source     : tatsu-lab/alpaca (HuggingFace)\n"
        f"  n_each           : {actual_n}\n"
        f"  seed             : {seed}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a balanced injection/clean dataset for benchmarking."
    )
    parser.add_argument(
        "--n-each",
        type=int,
        default=500,
        help="Samples per class (injection and clean). Default: 500 → 1000 total.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--output",
        default="data/benchmark_dataset.jsonl",
        help="Output JSONL path. Default: data/benchmark_dataset.jsonl",
    )
    args = parser.parse_args()
    build(n_each=args.n_each, seed=args.seed, output=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
