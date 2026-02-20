"""
gateway/security package

Two-tier prompt injection detection pipeline.

Tier 1 (fast path):  pattern_guard.py — regex / keyword matching against a
                     curated list of known injection patterns.  Runs in < 1 ms.
Tier 2 (ML path):   ml_guard.py — DistilBERT sequence classifier fine-tuned to
                     distinguish benign prompts from injection attempts.

guard.py orchestrates both tiers and exposes the single public interface:
    from gateway.security.guard import SecurityGuard
    result = await guard.scan(prompt_text)
"""
