"""
scripts/benchmark_security.py

Three-model prompt-injection benchmark.

Compares:
  ProtectAI-DeBERTa-v3  — ProtectAI/deberta-v3-base-prompt-injection-v2
  Meta-PromptGuard-86M  — meta-llama/Llama-Prompt-Guard-2-86M  (requires HF login + Llama license)
  Meta-PromptGuard-22M  — meta-llama/Llama-Prompt-Guard-2-22M  (requires HF login + Llama license)

Dataset:
  Build with scripts/build_benchmark_dataset.py first.
  Expected format: one JSON object per line, fields "text" (str) and "label" (0=clean, 1=injection).

Notes on Meta models:
  These models are gated on HuggingFace. Before running with --all-models you must:
    huggingface-cli login
  and accept the Llama license at:
    https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M
    https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M

Usage::

    # Build dataset first (one-time)
    python scripts/build_benchmark_dataset.py

    # Benchmark ProtectAI model only (no HF login needed)
    python scripts/benchmark_security.py --dataset data/benchmark_dataset.jsonl

    # Benchmark all three models (requires HF login + Llama license)
    python scripts/benchmark_security.py --dataset data/benchmark_dataset.jsonl --all-models

    # Generate PR curve plot in addition to table
    python scripts/benchmark_security.py --dataset data/benchmark_dataset.jsonl --pr-curve

    # Custom threshold and output dir
    python scripts/benchmark_security.py \\
        --dataset data/benchmark_dataset.jsonl \\
        --threshold 0.6 \\
        --output-dir results

Output files (written to --output-dir, default: results/):
  benchmark_results.json  — full per-model metrics + PR curve data
  pr_curve.png            — Precision-Recall curve (only with --pr-curve)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_CORE_MODELS: dict[str, str] = {
    "ProtectAI-DeBERTa-v3": "ProtectAI/deberta-v3-base-prompt-injection-v2",
}

_ALL_MODELS: dict[str, str] = {
    "ProtectAI-DeBERTa-v3": "ProtectAI/deberta-v3-base-prompt-injection-v2",
    "Meta-PromptGuard-86M": "meta-llama/Llama-Prompt-Guard-2-86M",
    "Meta-PromptGuard-22M": "meta-llama/Llama-Prompt-Guard-2-22M",
}

_WARMUP_N = 50       # samples to run before recording latency (JIT warm-up)
_BATCH_SIZE = 32     # inference batch size


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    """
    Load a JSONL benchmark dataset.

    Each line must be: {"text": "...", "label": 0_or_1}

    Args:
        path: Path to the JSONL file.

    Returns:
        List of {"text": str, "label": int} dicts.

    Raises:
        SystemExit if the file is missing or malformed.
    """
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Dataset not found: {path}", file=sys.stderr)
        print(
            "  Run: python scripts/build_benchmark_dataset.py",
            file=sys.stderr,
        )
        sys.exit(1)

    samples: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[ERROR] Line {lineno}: invalid JSON — {exc}", file=sys.stderr)
                sys.exit(1)
            if "text" not in obj or "label" not in obj:
                print(
                    f"[ERROR] Line {lineno}: missing 'text' or 'label' field.",
                    file=sys.stderr,
                )
                sys.exit(1)
            samples.append({"text": str(obj["text"]), "label": int(obj["label"])})

    if not samples:
        print(f"[ERROR] Dataset is empty: {path}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Loaded {len(samples)} samples "
        f"({sum(1 for s in samples if s['label'] == 1)} injection, "
        f"{sum(1 for s in samples if s['label'] == 0)} clean)"
    )
    return samples


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

def _is_injection(label: str) -> bool:
    """
    Return True when a pipeline output label indicates prompt injection.

    Handles label conventions across models:
      ProtectAI DeBERTa-v3  : "INJECTION" / "SAFE"
      Meta PromptGuard       : "INJECTION" / "BENIGN"  (or LABEL_1/LABEL_0 in some versions)
    """
    upper = label.upper()
    return any(kw in upper for kw in ("INJECTION", "MALICIOUS", "LABEL_1", "HARMFUL"))


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    model_name: str,
    model_id: str,
    samples: list[dict],
    threshold: float = 0.5,
) -> dict:
    """
    Load one model, warm it up, run inference on all samples, return metrics.

    Args:
        model_name: Display name (e.g. "ProtectAI-DeBERTa-v3").
        model_id:   HuggingFace repo ID.
        samples:    List of {"text": str, "label": int}.
        threshold:  Injection probability threshold (0–1).

    Returns:
        Dict with keys:
          model_name, model_id, threshold, n_samples,
          precision, recall, f1,
          tp, fp, tn, fn,
          latency_p50_ms, latency_p95_ms, latency_p99_ms,
          throughput_rps, total_time_s.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        print("transformers not installed — run: pip install transformers", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Model : {model_name}")
    print(f"  HF ID : {model_id}")
    print(f"{'='*60}")

    print("  Loading model …", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        pipe = hf_pipeline(
            "text-classification",
            model=model_id,
            truncation=True,
            max_length=512,
            device=-1,         # CPU; set device=0 for GPU
        )
    except Exception as exc:
        print(f"\n[ERROR] Failed to load {model_id}: {exc}", file=sys.stderr)
        raise
    load_time = time.perf_counter() - t0
    print(f"done ({load_time:.1f}s)")

    texts = [s["text"] for s in samples]
    labels_true = [s["label"] for s in samples]

    # ------------------------------------------------------------------
    # Warm up: run _WARMUP_N samples without recording latency.
    # This prevents cold-start JIT compilation from inflating p99.
    # ------------------------------------------------------------------
    warmup_texts = texts[:_WARMUP_N]
    print(f"  Warming up ({_WARMUP_N} samples) …", end=" ", flush=True)
    for _ in pipe(warmup_texts, batch_size=_BATCH_SIZE):
        pass
    print("done")

    # ------------------------------------------------------------------
    # Inference loop — time each sample individually for percentiles.
    # Batch inference still used for speed, but we measure wall-clock
    # end-to-end per sample by timing individual calls of batch size 1.
    # ------------------------------------------------------------------
    print(f"  Benchmarking {len(samples)} samples …", end=" ", flush=True)
    latencies_ms: list[float] = []
    labels_pred: list[int] = []

    t_total_start = time.perf_counter()
    for text in texts:
        t_start = time.perf_counter()
        result = pipe(text, truncation=True, max_length=512)[0]
        t_end = time.perf_counter()

        latencies_ms.append((t_end - t_start) * 1000.0)

        top_label: str = result["label"]
        top_score: float = result["score"]

        # Convert to injection probability regardless of which label is "top"
        if _is_injection(top_label):
            injection_prob = top_score
        else:
            injection_prob = 1.0 - top_score

        labels_pred.append(1 if injection_prob >= threshold else 0)

    total_time_s = time.perf_counter() - t_total_start
    print("done")

    # ------------------------------------------------------------------
    # Metrics — manual computation (avoids sklearn import requirement
    # at module level; sklearn is only needed when this function runs).
    # ------------------------------------------------------------------
    try:
        from sklearn.metrics import precision_recall_fscore_support
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels_true, labels_pred, pos_label=1, average="binary", zero_division=0
        )
    except ImportError:
        # Fallback: manual computation
        tp = sum(1 for t, p in zip(labels_true, labels_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(labels_true, labels_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(labels_true, labels_pred) if t == 1 and p == 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    tp = sum(1 for t, p in zip(labels_true, labels_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(labels_true, labels_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(labels_true, labels_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(labels_true, labels_pred) if t == 1 and p == 0)

    arr = np.array(latencies_ms)
    p50  = float(np.percentile(arr, 50))
    p95  = float(np.percentile(arr, 95))
    p99  = float(np.percentile(arr, 99))
    rps  = len(samples) / total_time_s if total_time_s > 0 else 0.0

    return {
        "model_name":      model_name,
        "model_id":        model_id,
        "threshold":       threshold,
        "n_samples":       len(samples),
        "precision":       float(prec),
        "recall":          float(rec),
        "f1":              float(f1),
        "tp":              tp,
        "fp":              fp,
        "tn":              tn,
        "fn":              fn,
        "latency_p50_ms":  p50,
        "latency_p95_ms":  p95,
        "latency_p99_ms":  p99,
        "throughput_rps":  rps,
        "total_time_s":    total_time_s,
    }


# ---------------------------------------------------------------------------
# PR curve
# ---------------------------------------------------------------------------

def compute_pr_curve(
    model_name: str,
    model_id: str,
    samples: list[dict],
    thresholds: list[float] | None = None,
) -> list[dict]:
    """
    Run inference once, then sweep thresholds to build a PR curve.

    Args:
        model_name: Display name.
        model_id:   HuggingFace repo ID.
        samples:    Labelled test records.
        thresholds: Threshold values to sweep (default: 0.1 to 0.99 in 0.05 steps).

    Returns:
        List of {"threshold": float, "precision": float, "recall": float, "f1": float}.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 1.00, 0.05).tolist()]

    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        print("transformers not installed.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  PR curve — {model_name} ({len(thresholds)} thresholds) …", end=" ", flush=True)
    pipe = hf_pipeline(
        "text-classification",
        model=model_id,
        truncation=True,
        max_length=512,
        device=-1,
    )

    # Single inference pass — collect raw injection probabilities
    texts = [s["text"] for s in samples]
    labels_true = [s["label"] for s in samples]

    injection_probs: list[float] = []
    for result in pipe(texts, batch_size=_BATCH_SIZE, truncation=True, max_length=512):
        top_label = result["label"]
        top_score = result["score"]
        prob = top_score if _is_injection(top_label) else 1.0 - top_score
        injection_probs.append(prob)

    print("done")

    curve: list[dict] = []
    for thr in thresholds:
        preds = [1 if p >= thr else 0 for p in injection_probs]
        tp = sum(1 for t, p in zip(labels_true, preds) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(labels_true, preds) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(labels_true, preds) if t == 1 and p == 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        curve.append({"threshold": thr, "precision": prec, "recall": rec, "f1": f1})

    return curve


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    """Print a formatted ASCII comparison table to stdout."""
    col_w = max(22, max(len(r["model_name"]) for r in results) + 2)
    label_w = 18

    sep = "+" + "-" * (label_w + 2) + ("+" + "-" * (col_w + 2)) * len(results) + "+"

    def row(label: str, values: list[str]) -> str:
        cells = "".join(f" {v:<{col_w}} |" for v in values)
        return f"| {label:<{label_w}} |{cells}"

    print("\n" + sep)
    print(row("Metric", [r["model_name"] for r in results]))
    print(sep)
    print(row("Precision",    [f"{r['precision']:.4f}"      for r in results]))
    print(row("Recall",       [f"{r['recall']:.4f}"         for r in results]))
    print(row("F1",           [f"{r['f1']:.4f}"             for r in results]))
    print(row("TP / FP",      [f"{r['tp']} / {r['fp']}"     for r in results]))
    print(row("TN / FN",      [f"{r['tn']} / {r['fn']}"     for r in results]))
    print(row("p50 latency",  [f"{r['latency_p50_ms']:.1f} ms" for r in results]))
    print(row("p95 latency",  [f"{r['latency_p95_ms']:.1f} ms" for r in results]))
    print(row("p99 latency",  [f"{r['latency_p99_ms']:.1f} ms" for r in results]))
    print(row("Throughput",   [f"{r['throughput_rps']:.1f} rps" for r in results]))
    print(row("Threshold",    [f"{r['threshold']}"           for r in results]))
    print(sep)
    print(
        "  Note: sequential single-sample CPU inference. Each sample timed individually\n"
        "  with time.perf_counter() after a 50-sample warmup. In production,\n"
        "  run_in_executor dispatches concurrent requests to a thread pool — per-request\n"
        "  latency is the same, but multiple requests can be in-flight simultaneously.\n"
        "  For lower latency: GPU ~3ms; ONNX + int8 quantization ~15-20ms on CPU."
    )
    print()


def plot_pr_curve(curve_data: dict[str, list[dict]], output_dir: str) -> None:
    """
    Save a Precision-Recall curve PNG to output_dir/pr_curve.png.

    Args:
        curve_data:  {model_name: [{"threshold", "precision", "recall", "f1"}, …]}
        output_dir:  Directory to write the PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping PR curve plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, curve in curve_data.items():
        recalls    = [pt["recall"]    for pt in curve]
        precisions = [pt["precision"] for pt in curve]
        ax.plot(recalls, precisions, marker="o", markersize=3, label=model_name)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Prompt Injection Classifiers")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(output_dir) / "pr_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PR curve saved to {out}")


def save_results(
    results: list[dict],
    curve_data: dict[str, list[dict]],
    output_dir: str,
) -> None:
    """
    Write full benchmark results to results/benchmark_results.json.

    Also prints a copy-pasteable markdown table to stdout.

    Args:
        results:    Per-model metric dicts from run_benchmark().
        curve_data: Per-model PR curve data (may be empty).
        output_dir: Directory to write the JSON file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": results,
        "pr_curves": curve_data,
    }
    out_file = out_dir / "benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {out_file}")

    # Markdown table for pasting into CONTEXT.md / README
    print("\n### Markdown table (paste into CONTEXT.md)\n")
    headers = ["Model", "Precision", "Recall", "F1", "p50 ms", "p99 ms", "RPS"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for r in results:
        print(
            f"| {r['model_name']} "
            f"| {r['precision']:.4f} "
            f"| {r['recall']:.4f} "
            f"| {r['f1']:.4f} "
            f"| {r['latency_p50_ms']:.1f} "
            f"| {r['latency_p99_ms']:.1f} "
            f"| {r['throughput_rps']:.1f} |"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-model prompt-injection benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        default="data/benchmark_dataset.jsonl",
        help="Path to JSONL dataset (default: data/benchmark_dataset.jsonl). "
             "Build with scripts/build_benchmark_dataset.py",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Injection probability threshold (0–1, default: 0.5).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Also benchmark Meta PromptGuard 86M and 22M (requires HF login + Llama license).",
    )
    parser.add_argument(
        "--pr-curve",
        action="store_true",
        help="Generate PR curve data and save pr_curve.png.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to write results JSON and optional PR curve PNG (default: results/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Full benchmark pipeline:
      1. Load dataset.
      2. Run benchmark for each selected model.
      3. Print comparison table.
      4. Optionally generate PR curves.
      5. Save JSON results + markdown table.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code 0 on success, 1 on any error.
    """
    args = _parse_args(argv)

    samples = load_dataset(args.dataset)

    model_registry = _ALL_MODELS if args.all_models else _CORE_MODELS

    if args.all_models:
        print(
            "\n[INFO] --all-models enabled. Meta PromptGuard models require:\n"
            "  huggingface-cli login\n"
            "  Accept Llama license at https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M\n"
            "  Accept Llama license at https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M\n"
        )

    results: list[dict] = []
    for model_name, model_id in model_registry.items():
        try:
            result = run_benchmark(model_name, model_id, samples, threshold=args.threshold)
            results.append(result)
        except Exception as exc:
            print(f"[ERROR] {model_name} failed: {exc}", file=sys.stderr)
            print("  Skipping this model and continuing.", file=sys.stderr)

    if not results:
        print("[ERROR] No models completed successfully.", file=sys.stderr)
        return 1

    print_table(results)

    curve_data: dict[str, list[dict]] = {}
    if args.pr_curve:
        for model_name, model_id in model_registry.items():
            if any(r["model_name"] == model_name for r in results):
                try:
                    curve_data[model_name] = compute_pr_curve(model_name, model_id, samples)
                except Exception as exc:
                    print(f"[WARN] PR curve failed for {model_name}: {exc}", file=sys.stderr)
        if curve_data:
            plot_pr_curve(curve_data, args.output_dir)

    save_results(results, curve_data, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
