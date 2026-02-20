"""
scripts/train_classifier.py

Standalone script for fine-tuning a DistilBERT sequence classifier to detect
prompt injection attempts.

This script is run once (or periodically as new training data is collected)
to produce the model checkpoint loaded by gateway/security/ml_guard.py.

Responsibilities:
  - Load a labelled dataset of benign and injection prompts from a CSV or JSONL
    file (label 0 = benign, 1 = injection).
  - Split into train / validation / test sets with stratified sampling.
  - Tokenize the prompts using DistilBERT's tokenizer.
  - Fine-tune a DistilBertForSequenceClassification model using the Hugging Face
    Trainer API with appropriate hyperparameters.
  - Evaluate the final model on the held-out test set and print precision, recall,
    F1, and a confusion matrix.
  - Save the fine-tuned model and tokenizer to the output directory specified by
    --output-dir (default: models/injection_classifier).
  - Log training metrics to stdout (and optionally to W&B if available).

Usage::

    python scripts/train_classifier.py \
        --data-path data/injection_dataset.jsonl \
        --output-dir models/injection_classifier \
        --epochs 3 \
        --batch-size 16 \
        --learning-rate 2e-5

Key functions:
  - main()                  — CLI entry point, orchestrates the pipeline
  - load_dataset(path)      — load and validate labelled data from disk
  - build_splits(data)      — stratified train/val/test split
  - tokenize(examples, tok) — batch tokenization for Trainer
  - compute_metrics(pred)   — sklearn precision/recall/F1 for Trainer callback
  - train(model, datasets, args) — configure and run the Trainer
  - evaluate(model, tok, test_data) — final evaluation pass with full report
  - save_model(model, tok, output_dir) — persist checkpoint to disk
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI entry point.

    Parses arguments, orchestrates data loading → splitting → training →
    evaluation → saving.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    args = _parse_args(argv)
    # TODO: implement pipeline
    ...
    return 0


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argument namespace with attributes:
          data_path, output_dir, epochs, batch_size, learning_rate,
          val_split, test_split, seed, fp16, push_to_hub.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for prompt injection detection"
    )
    parser.add_argument("--data-path", required=True, help="Path to labelled JSONL dataset")
    parser.add_argument("--output-dir", default="models/injection_classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable mixed-precision training")
    # TODO: add --push-to-hub flag for HF Hub upload
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    """
    Load a labelled prompt injection dataset from a JSONL file.

    Expected format (one JSON object per line)::

        {"text": "What is 2+2?", "label": 0}
        {"text": "Ignore all previous instructions and ...", "label": 1}

    Args:
        path: Path to the JSONL dataset file.

    Returns:
        List of dicts with "text" (str) and "label" (int: 0 or 1).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any record is missing required fields or has invalid label.
    """
    # TODO: implement
    ...


def build_splits(
    data: list[dict],
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split the dataset into train, validation, and test subsets.

    Uses stratified sampling to preserve the class ratio in each split.

    Args:
        data:       Full labelled dataset list.
        val_split:  Fraction of data for validation.
        test_split: Fraction of data for testing.
        seed:       Random seed for reproducibility.

    Returns:
        A 3-tuple: (train_data, val_data, test_data).
    """
    # TODO: implement — sklearn.model_selection.train_test_split with stratify
    ...


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize(examples: dict, tokenizer) -> dict:
    """
    Batch tokenization function compatible with the HF Datasets .map() API.

    Args:
        examples: A batch dict with "text" key (list of strings).
        tokenizer: The loaded DistilBERT tokenizer.

    Returns:
        Dict with input_ids, attention_mask (and token_type_ids if applicable).
    """
    # TODO: implement — tokenizer(examples["text"], truncation=True, padding=True)
    ...


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred) -> dict:
    """
    Compute precision, recall, F1, and accuracy for the Trainer evaluation loop.

    Compatible with transformers.Trainer's compute_metrics callback signature.

    Args:
        pred: EvalPrediction namedtuple with .predictions and .label_ids.

    Returns:
        Dict with keys: precision, recall, f1, accuracy.
    """
    # TODO: implement — sklearn.metrics.precision_recall_fscore_support, accuracy_score
    ...


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, datasets: dict, training_args) -> None:
    """
    Configure and run the Hugging Face Trainer.

    Args:
        model:         DistilBertForSequenceClassification instance.
        datasets:      Dict with "train" and "validation" HF Dataset objects.
        training_args: TrainingArguments instance.
    """
    # TODO: implement — Trainer(model, args, train_dataset, eval_dataset, compute_metrics)
    ...


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, test_data: list[dict]) -> dict:
    """
    Run a full evaluation pass on the test set and print a classification report.

    Args:
        model:     Fine-tuned model in eval mode.
        tokenizer: Paired tokenizer.
        test_data: Held-out test records.

    Returns:
        Dict with precision, recall, f1, accuracy, and confusion_matrix.
    """
    # TODO: implement — batch inference, sklearn classification_report
    ...


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, tokenizer, output_dir: str) -> None:
    """
    Save the fine-tuned model and tokenizer to disk.

    Also writes a config.json sidecar with training metadata (date, dataset
    path, hyperparameters) that MLGuard reads to populate model_version.

    Args:
        model:      Fine-tuned DistilBERT model.
        tokenizer:  Paired tokenizer.
        output_dir: Directory to write the checkpoint files.
    """
    # TODO: implement — model.save_pretrained(), tokenizer.save_pretrained(), write metadata
    ...


# ---------------------------------------------------------------------------
# CLI guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
