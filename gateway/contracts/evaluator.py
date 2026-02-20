"""
gateway/contracts/evaluator.py

Evaluates a single behavioral contract rule against a (request, response) pair.

Three evaluation tiers with different cost profiles:

  Deterministic (< 1ms):
    KeywordEvaluator, RegexEvaluator, LengthLimitEvaluator,
    LanguageMatchEvaluator, SchemaEvaluator

  Classifier (~10-15ms):
    SentimentEvaluator, TopicBoundaryEvaluator

  LLM Judge (~100-300ms):
    LLMJudgeEvaluator

Every evaluator returns an EvaluationResult with a compliance_score (0.0-1.0)
for drift detection.  Deterministic checks return 1.0 (pass) or 0.0 (fail).
Classifier and LLM judge checks return continuous scores.
"""

from __future__ import annotations

import asyncio
import json as _json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from gateway.contracts.schema import (
    AnyContract,
    CompositeContract,
    ContractAction,
    ContractType,
    EvaluationTier,
    KeywordContract,
    LanguageMatchContract,
    LengthLimitContract,
    LLMJudgeContract,
    RegexContract,
    SchemaContract,
    SentimentContract,
    TopicBoundaryContract,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ViolationDetail:
    contract_id: str
    message: str
    evidence: Optional[str] = None
    location: Optional[str] = None


@dataclass
class EvaluationResult:
    """
    Output of a single contract evaluation.

    compliance_score is the key addition for drift detection: a continuous
    0.0-1.0 value where 1.0 = fully compliant and 0.0 = clear violation.
    Deterministic checks produce binary 0/1; classifiers and LLM judges
    produce continuous values.
    """
    passed: bool
    action: Optional[ContractAction]
    contract_id: str
    evaluation_tier: EvaluationTier = EvaluationTier.DETERMINISTIC
    compliance_score: float = 1.0
    latency_ms: float = 0.0
    violation: Optional[ViolationDetail] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_response_text(response: dict) -> str:
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _resolve_target_field(response: dict, target_field: Optional[str]) -> Any:
    if not target_field:
        return response
    obj: Any = response
    for part in target_field.split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]
        else:
            obj = obj[part]
    return obj


def _extract_user_text(request: dict) -> str:
    """Extract the last user message from the request for language detection."""
    messages = request.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
                return " ".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Evaluator protocol
# ---------------------------------------------------------------------------

class ContractEvaluator(Protocol):
    async def evaluate(
        self,
        contract: AnyContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        ...


# ---------------------------------------------------------------------------
# Deterministic evaluators (< 1ms)
# ---------------------------------------------------------------------------

class KeywordEvaluator:
    async def evaluate(
        self,
        contract: KeywordContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        start = time.perf_counter()
        text = _extract_response_text(response)
        search_text = text if contract.case_sensitive else text.lower()

        for keyword in contract.keywords:
            needle = keyword if contract.case_sensitive else keyword.lower()

            if contract.match_whole_word:
                pattern = r"\b" + re.escape(needle) + r"\b"
                found = re.search(pattern, search_text) is not None
            else:
                found = needle in search_text

            if found:
                return EvaluationResult(
                    passed=False,
                    action=contract.action,
                    contract_id=contract.id,
                    evaluation_tier=EvaluationTier.DETERMINISTIC,
                    compliance_score=0.0,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    violation=ViolationDetail(
                        contract_id=contract.id,
                        message=f"Response contains forbidden keyword: {keyword!r}",
                        evidence=keyword,
                        location="choices[0].message.content",
                    ),
                )

        return EvaluationResult(
            passed=True, action=None, contract_id=contract.id,
            evaluation_tier=EvaluationTier.DETERMINISTIC,
            compliance_score=1.0,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


class RegexEvaluator:
    async def evaluate(
        self,
        contract: RegexContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        start = time.perf_counter()
        text = _extract_response_text(response)

        combined_flags = 0
        for flag_name in contract.flags:
            flag = getattr(re, flag_name.upper(), None)
            if flag is not None:
                combined_flags |= flag

        compiled = re.compile(contract.pattern, combined_flags)
        match = compiled.search(text)

        if match:
            return EvaluationResult(
                passed=False,
                action=contract.action,
                contract_id=contract.id,
                evaluation_tier=EvaluationTier.DETERMINISTIC,
                compliance_score=0.0,
                latency_ms=(time.perf_counter() - start) * 1000,
                violation=ViolationDetail(
                    contract_id=contract.id,
                    message=f"Response matches forbidden pattern: {contract.pattern!r}",
                    evidence=match.group(0)[:200],
                    location=f"choices[0].message.content[{match.start()}:{match.end()}]",
                ),
            )

        return EvaluationResult(
            passed=True, action=None, contract_id=contract.id,
            evaluation_tier=EvaluationTier.DETERMINISTIC,
            compliance_score=1.0,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


class LengthLimitEvaluator:
    """Evaluates word, character, and sentence count limits."""

    @staticmethod
    def _count_sentences(text: str) -> int:
        import re as _re
        sentences = _re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])

    async def evaluate(
        self,
        contract: LengthLimitContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        start = time.perf_counter()
        text = _extract_response_text(response)
        violations: list[str] = []

        if contract.max_words is not None:
            word_count = len(text.split())
            if word_count > contract.max_words:
                violations.append(f"Word count {word_count} exceeds limit {contract.max_words}")

        if contract.max_characters is not None:
            char_count = len(text)
            if char_count > contract.max_characters:
                violations.append(f"Character count {char_count} exceeds limit {contract.max_characters}")

        if contract.max_sentences is not None:
            sentence_count = self._count_sentences(text)
            if sentence_count > contract.max_sentences:
                violations.append(f"Sentence count {sentence_count} exceeds limit {contract.max_sentences}")

        elapsed = (time.perf_counter() - start) * 1000

        if violations:
            return EvaluationResult(
                passed=False,
                action=contract.action,
                contract_id=contract.id,
                evaluation_tier=EvaluationTier.DETERMINISTIC,
                compliance_score=0.0,
                latency_ms=elapsed,
                violation=ViolationDetail(
                    contract_id=contract.id,
                    message="; ".join(violations),
                    evidence=f"text_length={len(text)}",
                    location="choices[0].message.content",
                ),
            )

        return EvaluationResult(
            passed=True, action=None, contract_id=contract.id,
            evaluation_tier=EvaluationTier.DETERMINISTIC,
            compliance_score=1.0,
            latency_ms=elapsed,
        )


class LanguageMatchEvaluator:
    """
    Verify response language matches the user's message language.

    Uses langdetect which is fast (~1ms) and works for 55 languages.
    Falls open: if detection fails, the contract passes.
    """

    async def evaluate(
        self,
        contract: LanguageMatchContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        start = time.perf_counter()
        response_text = _extract_response_text(response)

        if not response_text.strip():
            return EvaluationResult(
                passed=True, action=None, contract_id=contract.id,
                evaluation_tier=EvaluationTier.DETERMINISTIC,
                compliance_score=1.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            from langdetect import detect as lang_detect  # type: ignore[import]
        except ImportError:
            return EvaluationResult(
                passed=True, action=None, contract_id=contract.id,
                evaluation_tier=EvaluationTier.DETERMINISTIC,
                compliance_score=1.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            if contract.expected_language:
                expected_lang = contract.expected_language
            else:
                user_text = _extract_user_text(request)
                if not user_text.strip():
                    return EvaluationResult(
                        passed=True, action=None, contract_id=contract.id,
                        evaluation_tier=EvaluationTier.DETERMINISTIC,
                        compliance_score=1.0,
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )
                expected_lang = lang_detect(user_text)

            detected_lang = lang_detect(response_text)
        except Exception:
            return EvaluationResult(
                passed=True, action=None, contract_id=contract.id,
                evaluation_tier=EvaluationTier.DETERMINISTIC,
                compliance_score=1.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        elapsed = (time.perf_counter() - start) * 1000
        matched = detected_lang == expected_lang

        if not matched:
            return EvaluationResult(
                passed=False,
                action=contract.action,
                contract_id=contract.id,
                evaluation_tier=EvaluationTier.DETERMINISTIC,
                compliance_score=0.0,
                latency_ms=elapsed,
                violation=ViolationDetail(
                    contract_id=contract.id,
                    message=f"Language mismatch: expected '{expected_lang}', detected '{detected_lang}'",
                    evidence=f"detected={detected_lang}, expected={expected_lang}",
                    location="choices[0].message.content",
                ),
            )

        return EvaluationResult(
            passed=True, action=None, contract_id=contract.id,
            evaluation_tier=EvaluationTier.DETERMINISTIC,
            compliance_score=1.0,
            latency_ms=elapsed,
        )


class SchemaEvaluator:
    async def evaluate(
        self,
        contract: SchemaContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        start = time.perf_counter()
        try:
            import jsonschema  # type: ignore[import]
        except ImportError:
            return EvaluationResult(
                passed=True, action=None, contract_id=contract.id,
                evaluation_tier=EvaluationTier.DETERMINISTIC,
                compliance_score=1.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            target = _resolve_target_field(response, contract.target_field)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return EvaluationResult(
                passed=False,
                action=contract.action,
                contract_id=contract.id,
                evaluation_tier=EvaluationTier.DETERMINISTIC,
                compliance_score=0.0,
                latency_ms=(time.perf_counter() - start) * 1000,
                violation=ViolationDetail(
                    contract_id=contract.id,
                    message=f"Could not extract target field {contract.target_field!r}: {exc}",
                    location=contract.target_field,
                ),
            )

        if isinstance(target, str):
            try:
                target = _json.loads(target)
            except _json.JSONDecodeError:
                pass

        try:
            jsonschema.validate(instance=target, schema=contract.json_schema)
        except jsonschema.ValidationError as exc:
            return EvaluationResult(
                passed=False,
                action=contract.action,
                contract_id=contract.id,
                evaluation_tier=EvaluationTier.DETERMINISTIC,
                compliance_score=0.0,
                latency_ms=(time.perf_counter() - start) * 1000,
                violation=ViolationDetail(
                    contract_id=contract.id,
                    message=f"JSON Schema validation failed: {exc.message}",
                    evidence=str(exc.instance)[:200],
                    location=contract.target_field or "response",
                ),
            )
        except jsonschema.SchemaError:
            return EvaluationResult(
                passed=True, action=None, contract_id=contract.id,
                evaluation_tier=EvaluationTier.DETERMINISTIC,
                compliance_score=1.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        return EvaluationResult(
            passed=True, action=None, contract_id=contract.id,
            evaluation_tier=EvaluationTier.DETERMINISTIC,
            compliance_score=1.0,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


# ---------------------------------------------------------------------------
# Classifier evaluators (~10-15ms)
# ---------------------------------------------------------------------------

class SentimentEvaluator:
    """
    Evaluates sentiment using a HuggingFace pipeline.
    Returns a continuous compliance_score for drift detection.
    """

    _pipeline = None
    _pipeline_model: str = ""

    async def evaluate(
        self,
        contract: SentimentContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        start = time.perf_counter()
        text = _extract_response_text(response)
        if not text:
            return EvaluationResult(
                passed=True, action=None, contract_id=contract.id,
                evaluation_tier=EvaluationTier.CLASSIFIER,
                compliance_score=1.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        loop = asyncio.get_event_loop()

        def _run_sentiment() -> float:
            if (
                SentimentEvaluator._pipeline is None
                or SentimentEvaluator._pipeline_model != contract.model
            ):
                from transformers import pipeline as hf_pipeline  # type: ignore[import]
                SentimentEvaluator._pipeline = hf_pipeline(
                    "sentiment-analysis", model=contract.model
                )
                SentimentEvaluator._pipeline_model = contract.model

            result = SentimentEvaluator._pipeline(text[:512])[0]
            label = result.get("label", "").upper()
            score = result.get("score", 0.5)

            if "POSITIVE" in label or "POS" in label:
                return score
            elif "NEGATIVE" in label or "NEG" in label:
                return -score
            else:
                return 0.0

        try:
            compound_score = await loop.run_in_executor(None, _run_sentiment)
        except Exception:
            return EvaluationResult(
                passed=True, action=None, contract_id=contract.id,
                evaluation_tier=EvaluationTier.CLASSIFIER,
                compliance_score=1.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Map the raw sentiment score to a 0-1 compliance score.
        # If min_score is 0.3 and compound is 0.5, compliance is high.
        # If compound is -0.2, compliance is low.
        if contract.min_score >= 0:
            compliance = max(0.0, min(1.0, (compound_score - contract.min_score + 1.0) / 2.0))
        else:
            compliance = 1.0 if compound_score >= contract.min_score else 0.0

        elapsed = (time.perf_counter() - start) * 1000
        passed = compound_score >= contract.min_score

        if not passed:
            return EvaluationResult(
                passed=False,
                action=contract.action,
                contract_id=contract.id,
                evaluation_tier=EvaluationTier.CLASSIFIER,
                compliance_score=compliance,
                latency_ms=elapsed,
                violation=ViolationDetail(
                    contract_id=contract.id,
                    message=(
                        f"Response sentiment score {compound_score:.3f} is below "
                        f"minimum threshold {contract.min_score:.3f}"
                    ),
                    evidence=f"sentiment_score={compound_score:.3f}",
                    location="choices[0].message.content",
                ),
            )

        return EvaluationResult(
            passed=True, action=None, contract_id=contract.id,
            evaluation_tier=EvaluationTier.CLASSIFIER,
            compliance_score=compliance,
            latency_ms=elapsed,
        )


class TopicBoundaryEvaluator:
    """
    Uses zero-shot classification (NLI-based) to check whether the response
    stays within allowed topics.

    The zero-shot approach means you define topics in plain English — no
    custom training data needed.  The model (default: facebook/bart-large-mnli)
    scores the response against each candidate label.
    """

    _classifier = None
    _classifier_model: str = ""

    async def evaluate(
        self,
        contract: TopicBoundaryContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        start = time.perf_counter()
        text = _extract_response_text(response)

        if not text.strip():
            return EvaluationResult(
                passed=True, action=None, contract_id=contract.id,
                evaluation_tier=EvaluationTier.CLASSIFIER,
                compliance_score=1.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        loop = asyncio.get_event_loop()

        def _run_classification() -> dict:
            if (
                TopicBoundaryEvaluator._classifier is None
                or TopicBoundaryEvaluator._classifier_model != contract.model
            ):
                from transformers import pipeline as hf_pipeline  # type: ignore[import]
                TopicBoundaryEvaluator._classifier = hf_pipeline(
                    "zero-shot-classification", model=contract.model
                )
                TopicBoundaryEvaluator._classifier_model = contract.model

            candidate_labels = list(contract.allowed_topics) + [contract.off_topic_label]
            result = TopicBoundaryEvaluator._classifier(
                text[:512],
                candidate_labels=candidate_labels,
            )
            return result

        try:
            result = await loop.run_in_executor(None, _run_classification)
        except Exception:
            return EvaluationResult(
                passed=True, action=None, contract_id=contract.id,
                evaluation_tier=EvaluationTier.CLASSIFIER,
                compliance_score=1.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        labels = result.get("labels", [])
        scores = result.get("scores", [])
        elapsed = (time.perf_counter() - start) * 1000

        if not labels:
            return EvaluationResult(
                passed=True, action=None, contract_id=contract.id,
                evaluation_tier=EvaluationTier.CLASSIFIER,
                compliance_score=1.0,
                latency_ms=elapsed,
            )

        top_label = labels[0]
        top_score = scores[0]

        # Compliance score: sum of scores for allowed topics
        allowed_set = set(t.lower() for t in contract.allowed_topics)
        compliance = sum(
            s for l, s in zip(labels, scores)
            if l.lower() in allowed_set
        )

        on_topic = (
            top_label.lower() in allowed_set
            and top_score >= contract.threshold
        )

        if not on_topic:
            return EvaluationResult(
                passed=False,
                action=contract.action,
                contract_id=contract.id,
                evaluation_tier=EvaluationTier.CLASSIFIER,
                compliance_score=compliance,
                latency_ms=elapsed,
                violation=ViolationDetail(
                    contract_id=contract.id,
                    message=(
                        f"Response classified as '{top_label}' ({top_score:.3f}) "
                        f"which is outside allowed topics: {contract.allowed_topics}"
                    ),
                    evidence=f"top_label={top_label}, score={top_score:.3f}",
                    location="choices[0].message.content",
                ),
            )

        return EvaluationResult(
            passed=True, action=None, contract_id=contract.id,
            evaluation_tier=EvaluationTier.CLASSIFIER,
            compliance_score=compliance,
            latency_ms=elapsed,
        )


# ---------------------------------------------------------------------------
# LLM Judge evaluator (~100-300ms)
# ---------------------------------------------------------------------------

class LLMJudgeEvaluator:
    """
    Uses a small LLM to evaluate complex natural-language assertions.

    The judge model is called through the gateway's own provider system so
    it benefits from the same routing, retry, and observability infrastructure.

    The judge prompt asks the model to return a JSON compliance score.
    Parsing failures are handled gracefully (fail open).
    """

    async def evaluate(
        self,
        contract: LLMJudgeContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        start = time.perf_counter()
        response_text = _extract_response_text(response)

        if not response_text.strip():
            return EvaluationResult(
                passed=True, action=None, contract_id=contract.id,
                evaluation_tier=EvaluationTier.LLM_JUDGE,
                compliance_score=1.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        judge_request = {
            "model": contract.judge_model,
            "messages": [
                {"role": "system", "content": contract.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Assertion: {contract.assertion}\n\n"
                        f"Response to evaluate:\n{response_text[:2000]}"
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 150,
        }

        try:
            score, reasoning = await self._call_judge(judge_request)
        except Exception:
            return EvaluationResult(
                passed=True, action=None, contract_id=contract.id,
                evaluation_tier=EvaluationTier.LLM_JUDGE,
                compliance_score=1.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        elapsed = (time.perf_counter() - start) * 1000
        passed = score >= contract.threshold

        if not passed:
            return EvaluationResult(
                passed=False,
                action=contract.action,
                contract_id=contract.id,
                evaluation_tier=EvaluationTier.LLM_JUDGE,
                compliance_score=score,
                latency_ms=elapsed,
                violation=ViolationDetail(
                    contract_id=contract.id,
                    message=(
                        f"LLM judge scored {score:.3f} (threshold {contract.threshold:.3f}): "
                        f"{reasoning}"
                    ),
                    evidence=f"score={score:.3f}, reasoning={reasoning[:200]}",
                    location="choices[0].message.content",
                ),
            )

        return EvaluationResult(
            passed=True, action=None, contract_id=contract.id,
            evaluation_tier=EvaluationTier.LLM_JUDGE,
            compliance_score=score,
            latency_ms=elapsed,
        )

    async def _call_judge(self, judge_request: dict) -> tuple[float, str]:
        """
        Call the judge model via the gateway's provider system.

        Uses a lazy import of ProviderRouter to avoid circular imports.
        Falls back to httpx direct call if the router isn't available.
        """
        try:
            from gateway.router import ProviderRouter
            from gateway.config import get_settings

            settings = get_settings()
            router = ProviderRouter(settings)
            routing = router.resolve(judge_request)
            adapter = routing.adapter_class(settings.providers)
            response_body = await adapter.complete(judge_request)
        except Exception:
            # Fallback: try calling via httpx to a local endpoint
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "http://localhost:8000/v1/chat/completions",
                    json=judge_request,
                    headers={"Authorization": "Bearer internal-judge"},
                )
                resp.raise_for_status()
                response_body = resp.json()

        content = response_body["choices"][0]["message"]["content"]
        return self._parse_judge_response(content)

    @staticmethod
    def _parse_judge_response(content: str) -> tuple[float, str]:
        """Parse the JSON score from the judge model's response."""
        content = content.strip()

        # Try direct JSON parse
        try:
            parsed = _json.loads(content)
            score = float(parsed.get("score", 0.5))
            reasoning = str(parsed.get("reasoning", ""))
            return max(0.0, min(1.0, score)), reasoning
        except (_json.JSONDecodeError, ValueError, AttributeError):
            pass

        # Try extracting JSON from markdown code block
        import re as _re
        json_match = _re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, _re.DOTALL)
        if json_match:
            try:
                parsed = _json.loads(json_match.group(1))
                score = float(parsed.get("score", 0.5))
                reasoning = str(parsed.get("reasoning", ""))
                return max(0.0, min(1.0, score)), reasoning
            except (_json.JSONDecodeError, ValueError):
                pass

        # Try finding a bare number
        num_match = _re.search(r'(\d+\.?\d*)', content)
        if num_match:
            score = float(num_match.group(1))
            if score > 1.0:
                score = score / 100.0
            return max(0.0, min(1.0, score)), content[:200]

        # Default: ambiguous, assume moderate compliance
        return 0.5, f"Could not parse judge response: {content[:200]}"


# ---------------------------------------------------------------------------
# Composite evaluator
# ---------------------------------------------------------------------------

class CompositeEvaluator:
    async def evaluate(
        self,
        contract: CompositeContract,
        request: dict,
        response: dict,
    ) -> EvaluationResult:
        start = time.perf_counter()
        sub_results: list[EvaluationResult] = await asyncio.gather(
            *[evaluate_contract(sub, request, response) for sub in contract.contracts]
        )

        failures = [r for r in sub_results if not r.passed]
        avg_compliance = (
            sum(r.compliance_score for r in sub_results) / len(sub_results)
            if sub_results else 1.0
        )

        # Determine the highest-cost tier among sub-contracts
        tier_order = {EvaluationTier.DETERMINISTIC: 0, EvaluationTier.CLASSIFIER: 1, EvaluationTier.LLM_JUDGE: 2}
        max_tier = max(sub_results, key=lambda r: tier_order.get(r.evaluation_tier, 0)).evaluation_tier if sub_results else EvaluationTier.DETERMINISTIC

        if contract.operator == "OR":
            if failures:
                return EvaluationResult(
                    passed=False,
                    action=contract.action,
                    contract_id=contract.id,
                    evaluation_tier=max_tier,
                    compliance_score=avg_compliance,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    violation=ViolationDetail(
                        contract_id=contract.id,
                        message=(
                            f"Composite (OR) violated: "
                            f"{len(failures)}/{len(sub_results)} sub-contracts failed"
                        ),
                        evidence=failures[0].violation.message if failures[0].violation else None,
                    ),
                )
        else:
            if len(failures) == len(sub_results):
                return EvaluationResult(
                    passed=False,
                    action=contract.action,
                    contract_id=contract.id,
                    evaluation_tier=max_tier,
                    compliance_score=avg_compliance,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    violation=ViolationDetail(
                        contract_id=contract.id,
                        message=(
                            f"Composite (AND) violated: "
                            f"all {len(sub_results)} sub-contracts failed"
                        ),
                    ),
                )

        return EvaluationResult(
            passed=True, action=None, contract_id=contract.id,
            evaluation_tier=max_tier,
            compliance_score=avg_compliance,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


# ---------------------------------------------------------------------------
# Dispatch table and top-level dispatcher
# ---------------------------------------------------------------------------

_EVALUATORS: dict[str, Any] = {
    ContractType.KEYWORD:        KeywordEvaluator(),
    ContractType.REGEX:          RegexEvaluator(),
    ContractType.LENGTH_LIMIT:   LengthLimitEvaluator(),
    ContractType.LANGUAGE_MATCH: LanguageMatchEvaluator(),
    ContractType.SENTIMENT:      SentimentEvaluator(),
    ContractType.TOPIC_BOUNDARY: TopicBoundaryEvaluator(),
    ContractType.LLM_JUDGE:      LLMJudgeEvaluator(),
    ContractType.SCHEMA:         SchemaEvaluator(),
    ContractType.COMPOSITE:      CompositeEvaluator(),
}


async def evaluate_contract(
    contract: AnyContract,
    request: dict,
    response: dict,
) -> EvaluationResult:
    """Route contract to the appropriate evaluator based on its type."""
    evaluator = _EVALUATORS.get(contract.type)
    if evaluator is None:
        raise ValueError(f"No evaluator registered for contract type: {contract.type!r}")
    return await evaluator.evaluate(contract, request, response)
