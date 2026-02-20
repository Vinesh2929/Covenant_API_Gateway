"""
gateway/security/ml_guard.py

Tier-2 prompt injection detection via a fine-tuned DistilBERT classifier.

Responsibilities:
  - Load a fine-tuned DistilBERT sequence classification model from a local
    directory (the model is trained by scripts/train_classifier.py).
  - Tokenize an input prompt and run a forward pass to obtain class
    probabilities: P(benign) and P(injection).
  - Compare P(injection) against a configurable confidence threshold and
    return a structured ScanResult.
  - Manage the model lifecycle: lazy loading on first call (to avoid slowing
    down startup), and explicit warm-up method called during the FastAPI
    lifespan hook.
  - Run inference on CPU by default; detect and use MPS (Apple Silicon) or
    CUDA if available.
  - Implement basic input length guardrails (truncation) to keep inference
    time predictable.

Why DistilBERT?  It offers ~97% of BERT accuracy at 40% of the parameter
count, making it practical for real-time gateway inference without a GPU.

Key classes / functions:
  - ScanResult               — dataclass: is_injection, confidence, label, latency_ms
  - MLGuard                  — main class
    - __init__(settings)     — store model path + threshold; model NOT loaded yet
    - warm_up()              — async: load model weights into memory
    - scan(text)             — async: tokenize → infer → return ScanResult
    - _load_model()          — internal: load tokenizer + model from disk
    - _infer(input_ids)      — internal: run model forward pass, return softmax probs
    - _select_device()       — internal: choose cuda / mps / cpu
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from gateway.config import SecuritySettings


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """
    Output of a single MLGuard.scan() call.

    Attributes:
        is_injection:  True when the model's P(injection) exceeds the threshold.
        confidence:    The model's probability score for the injection class (0–1).
        label:         Human-readable label: "INJECTION" or "BENIGN".
        latency_ms:    Wall-clock time taken for the inference pass.
        model_version: Identifier of the model checkpoint used (from config.json).
    """
    is_injection: bool
    confidence: float
    label: str
    latency_ms: float
    model_version: str = "unknown"


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

class MLGuard:
    """
    DistilBERT-based injection classifier (Tier 2).

    The model is expected to be a Hugging Face `AutoModelForSequenceClassification`
    checkpoint saved to disk by scripts/train_classifier.py.  Labels must be:
      - 0 → BENIGN
      - 1 → INJECTION

    Usage::

        guard = MLGuard(settings.security)
        await guard.warm_up()               # call once at startup
        result = await guard.scan(prompt)
        if result.is_injection:
            raise HTTPException(400, detail="Injection detected by ML model")
    """

    def __init__(self, settings: SecuritySettings) -> None:
        """
        Store configuration. Model weights are NOT loaded here to keep app
        startup fast.  Call warm_up() to load the model explicitly.

        Args:
            settings: SecuritySettings slice containing ml_model_path and
                      ml_confidence_threshold.
        """
        self._settings = settings
        self._tokenizer = None   # set by _load_model()
        self._model = None       # set by _load_model()
        self._device: Optional[str] = None
        self._model_version: str = "unknown"

    async def warm_up(self) -> None:
        """
        Load the tokenizer and model weights into memory.

        Called once from the FastAPI lifespan startup hook so that the first
        real request does not pay the model-loading latency penalty.

        Logs a warning (but does not raise) if the model directory does not
        exist, allowing the gateway to start in "ML guard disabled" mode.
        """
        # TODO: implement — run _load_model() in a thread pool executor to avoid
        # blocking the event loop during weight loading
        ...

    async def scan(self, text: str) -> ScanResult:
        """
        Classify `text` as benign or injection using the DistilBERT model.

        If the model has not been loaded (warm_up not yet called or model path
        missing), returns a ScanResult with is_injection=False and a note in
        the label field.

        Args:
            text: The prompt string to classify.  Will be truncated to
                  max_length=512 tokens if necessary.

        Returns:
            A ScanResult with the classification decision and metadata.
        """
        # TODO: implement
        ...

    def _load_model(self) -> None:
        """
        Synchronous model loading — should be called inside a thread pool to
        avoid blocking the async event loop.

        Steps:
          1. Detect the compute device via _select_device().
          2. Load the tokenizer with AutoTokenizer.from_pretrained().
          3. Load the model with AutoModelForSequenceClassification.from_pretrained().
          4. Move the model to the selected device and set eval mode.
          5. Read the model version from config.json if present.
        """
        # TODO: implement
        ...

    def _infer(self, text: str) -> tuple[float, float]:
        """
        Run a single forward pass and return class probabilities.

        Args:
            text: The (already truncation-guarded) prompt string.

        Returns:
            A 2-tuple: (p_benign, p_injection) as floats summing to ~1.0.
        """
        # TODO: implement — tokenize, run model, softmax
        ...

    def _select_device(self) -> str:
        """
        Choose the best available compute device.

        Priority: CUDA GPU → Apple MPS → CPU.

        Returns:
            A torch device string: "cuda", "mps", or "cpu".
        """
        # TODO: implement using torch.cuda.is_available() / torch.backends.mps.is_available()
        ...
