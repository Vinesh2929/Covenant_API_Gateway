"""
gateway/security/ml_guard.py

Tier-2 prompt injection detection via a DeBERTa-v3 classifier.

MODEL STRATEGY — WHY DeBERTa, NOT DistilBERT FROM SCRATCH?
  Rather than fine-tuning DistilBERT from scratch (which produces a model
  that already exists publicly), Covenant uses ProtectAI's
  `deberta-v3-base-prompt-injection-v2` as its baseline Tier-2 classifier.
  This is the same model real production systems use.

  Covenant's contribution is:
    1. Benchmarking the three leading open-source prompt injection classifiers
       against each other on the same evaluation set.
    2. Fine-tuning the winner on domain-specific data for the deployment
       environment (e.g. code-heavy prompts, multi-turn conversations).
    3. Integrating the chosen model into a two-tier pipeline where the
       fast regex guard (Tier 1) short-circuits HIGH/CRITICAL patterns,
       and the ML model (Tier 2) handles the ambiguous middle ground.

BENCHMARK COMPARISON (run via scripts/benchmark_security.py):

  Model                              Params   p50 CPU   p99 CPU   Precision  Recall   F1
  ─────────────────────────────────  ───────  ────────  ────────  ─────────  ──────   ──────
  ProtectAI deberta-v3-base-v2       184 M    ~73 ms    ~188 ms   1.000      0.429    0.600   ← default
  Meta Prompt Guard 2 86M            86 M     ~74 ms    ~129 ms   1.000      0.246    0.395
  Meta Prompt Guard 2 22M            22 M     ~32 ms    ~126 ms   1.000      0.212    0.350

  All measured: 406 samples (deepset/prompt-injections + Alpaca 50/50), 50-sample warmup,
  sequential single-sample inference on Apple M-series CPU. Zero false positives across all
  three models at every threshold tested (0.05–0.99).
  ONNX + int8 quantization would bring DeBERTa-v3 to ~15-20ms on CPU.

  Run all three on the same test set. Measure precision, recall, F1, and
  per-prompt CPU latency. The benchmark script writes a PR-curve plot and
  a JSON summary — use these to select the model for your deployment.

DEFAULT MODEL:
  settings.security.ml_model_path controls which checkpoint is used.
  The default value in .env.example points to a local copy of
  ProtectAI/deberta-v3-base-prompt-injection-v2 downloaded by
  `scripts/download_models.py`.  You can swap this path to point at a
  fine-tuned checkpoint or a different model without changing this file.

Responsibilities:
  - Load a Hugging Face AutoModelForSequenceClassification checkpoint
    (DeBERTa-v3 or compatible) from a local directory.
  - Tokenize an input prompt and run a forward pass to obtain class
    probabilities: P(benign) and P(injection).
  - Compare P(injection) against a configurable confidence threshold and
    return a structured ScanResult.
  - Manage the model lifecycle: lazy loading on first call (to avoid slowing
    down startup), and explicit warm-up method called during the FastAPI
    lifespan hook.
  - Run inference on CPU by default; detect and use MPS (Apple Silicon) or
    CUDA if available.
  - Implement basic input length guardrails (truncation).

Key classes / functions:
  - ScanResult               — dataclass: is_injection, confidence, label, latency_ms
  - MLGuard                  — main class
    - __init__(settings)     — store model path + threshold; model NOT loaded yet
    - warm_up()              — async: load model weights into memory
    - scan(text)             — async: tokenize → infer → return ScanResult
    - _load_model()          — internal: load tokenizer + model from disk
    - _infer(text)           — internal: run model forward pass, return softmax probs
    - _select_device()       — internal: choose cuda / mps / cpu
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
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
    DeBERTa-v3-based injection classifier (Tier 2).

    Uses ProtectAI/deberta-v3-base-prompt-injection-v2 as the default
    backbone — the same model widely deployed in production LLM gateways.

    The gateway supports any Hugging Face AutoModelForSequenceClassification
    checkpoint; swap the model path in settings to use Meta Prompt Guard 2
    or a domain-specific fine-tune without code changes.

    WHY DeBERTa-v3 OVER DistilBERT?
      DeBERTa-v3 uses disentangled attention (separate position and content
      attention matrices) + replaced-token detection pre-training, which gives
      it better contextual understanding than DistilBERT for subtle injection
      patterns (indirect injection, role-play jailbreaks, multi-step attacks).

      ProtectAI's benchmark shows deberta-v3-base-prompt-injection-v2 achieves
      ~98% precision at 95% recall on public injection datasets — substantially
      better than a DistilBERT fine-tune at the same threshold.

    DEFERRED LOADING:
      Model weights (~700 MB for DeBERTa-v3-base) are NOT loaded at object
      construction time.  They are loaded in warm_up(), called from the FastAPI
      lifespan hook.  This keeps import time fast and makes model loading
      failure a recoverable logged warning rather than a startup crash.

    Usage::

        guard = MLGuard(settings.security)
        await guard.warm_up()               # once at startup
        result = await guard.scan(prompt)
        if result.is_injection:
            raise HTTPException(400, "Injection detected by ML model")
    """

    # Maximum number of tokens sent to the model.
    # DeBERTa-v3-base was pre-trained with a 512-token context (same as BERT).
    # Longer inputs are silently truncated by the tokenizer.
    # 512 tokens ≈ 380 words — sufficient for any realistic single-turn prompt.
    _MAX_TOKENS: int = 512

    # Label index → human-readable string mapping.
    # ProtectAI's model uses: 0 → LABEL_0 (benign), 1 → LABEL_1 (injection).
    # We map these to clearer names for logging and trace metadata.
    _LABELS: dict[int, str] = {0: "BENIGN", 1: "INJECTION"}

    def __init__(self, settings: SecuritySettings) -> None:
        """
        Store configuration. Model weights are NOT loaded here.

        Args:
            settings: SecuritySettings slice with ml_model_path and
                      ml_confidence_threshold.
        """
        self._settings = settings
        self._tokenizer = None   # set by _load_model()
        self._model = None       # set by _load_model()
        self._device: Optional[str] = None
        self._model_version: str = "unknown"

        # Flag to avoid trying to reload a missing model on every scan() call.
        self._load_attempted: bool = False

    async def warm_up(self) -> None:
        """
        Load the tokenizer and model weights into memory.

        Runs the blocking I/O (reading ~700 MB from disk) inside asyncio's
        default thread pool executor so the event loop stays responsive.

        Why run_in_executor?
          `asyncio.get_event_loop().run_in_executor(None, fn)` runs `fn` in a
          background thread from Python's default ThreadPoolExecutor.  This
          prevents the single-threaded event loop from freezing for the 3-8
          seconds that DeBERTa model loading takes on a cold start.

        Graceful failure:
          If the model directory doesn't exist (e.g. you haven't run
          scripts/download_models.py yet), we log a warning but don't raise.
          The guard will then return is_injection=False on every scan(),
          effectively disabling Tier 2 until the model is available.
        """
        if self._load_attempted:
            return  # idempotent — safe to call multiple times

        self._load_attempted = True

        import asyncio
        import logging

        loop = asyncio.get_event_loop()

        try:
            # _load_model() is synchronous (PyTorch / HuggingFace are not async).
            # run_in_executor moves it to a thread so the event loop isn't blocked.
            await loop.run_in_executor(None, self._load_model)
        except FileNotFoundError:
            # Model not yet downloaded. This is expected on a fresh install.
            # Run: python scripts/download_models.py
            logging.getLogger(__name__).warning(
                "ML injection classifier not found at '%s'. "
                "Tier-2 detection disabled. "
                "Run scripts/download_models.py to fetch the model.",
                self._settings.ml_model_path,
            )

    async def scan(self, text: str) -> ScanResult:
        """
        Classify `text` as benign or injection using the DeBERTa-v3 model.

        If the model is not loaded (missing checkpoint or warm_up not called),
        returns a safe ScanResult(is_injection=False) so the gateway degrades
        gracefully to pattern-only scanning.

        Args:
            text: The full conversation as a single string (from _extract_prompt_text
                  in main.py).  Truncated to _MAX_TOKENS tokens automatically.

        Returns:
            A ScanResult with the classification decision and inference latency.
        """
        # Degrade gracefully if model failed to load.
        if self._model is None or self._tokenizer is None:
            return ScanResult(
                is_injection=False,
                confidence=0.0,
                label="MODEL_UNAVAILABLE",
                latency_ms=0.0,
                model_version=self._model_version,
            )

        import asyncio
        loop = asyncio.get_event_loop()

        # Run inference in a thread pool (PyTorch forward pass is CPU-bound).
        # This prevents blocking the event loop during matrix multiplications.
        t0 = time.perf_counter()
        p_benign, p_injection = await loop.run_in_executor(None, self._infer, text)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Apply the confidence threshold.
        # At threshold=0.85: if the model is 85%+ confident this is an injection,
        # block it.  Lower threshold → higher recall (catch more injections) but
        # more false positives.  Tune in settings.ml_confidence_threshold.
        is_injection = p_injection >= self._settings.ml_confidence_threshold
        label = "INJECTION" if is_injection else "BENIGN"

        return ScanResult(
            is_injection=is_injection,
            confidence=p_injection,
            label=label,
            latency_ms=latency_ms,
            model_version=self._model_version,
        )

    def _load_model(self) -> None:
        """
        Synchronous model loader — always call inside run_in_executor.

        Loads the tokenizer and model from the local checkpoint directory.
        The checkpoint can be:
          - ProtectAI/deberta-v3-base-prompt-injection-v2 (default)
          - Meta's Prompt Guard 2 (86M or 22M variant)
          - A local fine-tune of any of the above

        Steps:
          1. Select the compute device (CUDA > MPS > CPU).
          2. Load the tokenizer (handles SentencePiece subword tokenisation).
          3. Load the AutoModelForSequenceClassification weights.
          4. Move the model to the device and set eval() mode.
             eval() disables dropout layers used only during training,
             making inference deterministic and slightly faster.
          5. Read the model version from config.json for traceability.

        Raises:
            FileNotFoundError: If ml_model_path does not exist on disk.
        """
        # Deferred imports — transformers and torch take ~500 ms to import.
        # By importing here (inside the thread), we don't pay this cost at
        # module load time, keeping `from gateway.main import app` fast.
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        import json as _json

        model_path = self._settings.ml_model_path

        # Raise early with a clear error if the path doesn't exist.
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"ML model directory not found: '{model_path}'. "
                "Run `python scripts/download_models.py` to download the model."
            )

        self._device = self._select_device()

        # Load the tokenizer.
        # DeBERTa-v3 uses a SentencePiece tokenizer (sp_model.model).
        # from_pretrained() reads tokenizer_config.json and the vocab files.
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load the model (weights from pytorch_model.bin or model.safetensors).
        # AutoModelForSequenceClassification auto-detects the DeBERTa architecture
        # from config.json and builds the correct PyTorch model class.
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Move model weights to the selected device.
        # On CPU this is a no-op; on GPU it copies tensors to VRAM.
        self._model = self._model.to(self._device)

        # eval() switches off training-specific layers (dropout, attention dropout).
        # Without this, the model gives slightly different outputs each call.
        self._model.eval()

        # Read the model version from config.json for logging/tracing.
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            config_data = _json.loads(config_path.read_text())
            # _name_or_path is set by Hugging Face to the hub model ID.
            self._model_version = config_data.get(
                "_name_or_path",
                config_data.get("_gateway_model_version", "unknown")
            )

    def _infer(self, text: str) -> tuple[float, float]:
        """
        Run a single inference pass and return (p_benign, p_injection).

        Always called from a thread pool — this is synchronous PyTorch code.

        Steps:
          1. Tokenize the text (produces input_ids + attention_mask tensors).
          2. Run the model's forward pass inside torch.no_grad() to skip
             gradient tracking (inference only — saves memory and time).
          3. Apply softmax to convert raw logits into probabilities.
          4. Return probabilities for class 0 (benign) and class 1 (injection).

        Args:
            text: The prompt string.  Truncated to _MAX_TOKENS by the tokenizer.

        Returns:
            (p_benign, p_injection) both in [0, 1] summing to ~1.0.
        """
        import torch

        # Tokenize.
        # return_tensors="pt"  → return PyTorch tensors, not Python lists.
        # truncation=True       → silently truncate to max_length tokens.
        # padding=True          → pad shorter sequences (needed for batches;
        #                         for single inputs it's a no-op).
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._MAX_TOKENS,
            padding=True,
        )

        # Move input tensors to the same device as the model.
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Forward pass.
        # torch.no_grad() tells PyTorch not to build a computation graph for
        # backpropagation — we're doing inference, not training, so this saves
        # significant memory and speeds up the forward pass.
        with torch.no_grad():
            outputs = self._model(**inputs)

        # outputs.logits is a tensor of shape (batch_size=1, num_labels=2).
        # Logits are raw, unnormalised scores.  Softmax converts them to
        # probabilities that sum to 1.0.
        logits = outputs.logits[0]          # shape: (2,) — remove batch dimension
        probs = torch.softmax(logits, dim=0)

        p_benign = probs[0].item()          # .item() converts tensor → Python float
        p_injection = probs[1].item()

        return p_benign, p_injection

    def _select_device(self) -> str:
        """
        Choose the best available compute device for PyTorch inference.

        Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.

        WHY THIS ORDER?
          CUDA is the fastest for transformer inference (10-50x over CPU).
          MPS (Metal Performance Shaders) is Apple's GPU acceleration for M1/M2/M3
          — faster than CPU but typically slower than a dedicated NVIDIA GPU.
          CPU is the safe fallback that always works.

          DeBERTa-v3-base on CPU: ~100 ms per prompt (sequential single-sample,
          measured; ONNX + int8 quantization brings this to ~15-20 ms on CPU).
          DeBERTa-v3-base on GPU: ~2-5 ms per prompt.

        Returns:
            A torch device string: "cuda", "mps", or "cpu".
        """
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # torch.backends.mps.is_available() was added in PyTorch 1.12.
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
