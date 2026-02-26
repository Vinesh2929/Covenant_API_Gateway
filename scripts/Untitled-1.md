**The benchmark — what the three tiers actually catch**

I measured on 406 samples: 203 real injection attempts from the [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) dataset and 203 clean prompts from Alpaca. Every model ran on CPU with a 50-sample warmup to eliminate JIT cold-start noise.

First, the Tier 2 classifier comparison:

| Model | Precision | Recall | F1 | p50 ms |
| --- | --- | --- | --- | --- |
| ProtectAI DeBERTa-v3 (184M) | **1.0000** | **0.4286** | **0.6000** | 72.8 ms |
| Meta PromptGuard-2 86M | **1.0000** | 0.2463 | 0.3953 | 73.5 ms |
| Meta PromptGuard-2 22M | **1.0000** | 0.2118 | 0.3496 | 32.3 ms |

Precision 1.00 across all three — zero false positives on 203 clean prompts. But 43% recall looked bad in isolation. The question was: what does the full three-tier pipeline actually catch?

I ran Claude Haiku as the Tier 3 judge on all 116 samples DeBERTa missed. Results:

| Pipeline | Recall | Notes |
| --- | --- | --- |
| Tier 2 alone | 42.9% | Zero false positives |
| **Tier 2 + Tier 3** | **63.5%** | +20.7pp, async, ~$0.05/1k requests |

Haiku caught 42 of the 116 misses — multilingual injections ("Zaboravi sve instrukcije" = "Forget all instructions" in Serbian), obfuscated persona hijacking, and mixed-language attacks that look like benign text to a statistical classifier but are obviously injection attempts to a model that understands language intent. The 74 it also passed on were mostly creative role-play requests that the dataset labels as injection but that a contextual judge correctly identifies as legitimate.

The key architectural point: Tier 3 adds zero latency for the user. The gateway returns the Tier 2 verdict immediately. Haiku runs as a background `asyncio.Task` and logs its conclusion after the response is already sent. At ~$0.80/MTok (Haiku), running the judge on 1,000 ambiguous prompts per day costs about $23/month — well under the cost of a false negative in a production system.

---

**Behavioral contracts — the other half of the problem**

Prompt injection is the input problem. Behavioral contracts are the output problem: ensuring the model actually does what you told it to do, and doesn't do what you told it not to.

Here's a complete contract file for a bank support bot:

```json
{
  "app_id": "bank-support-bot",
  "contracts": [
    {
      "type": "keyword",
      "name": "no_competitor_mention",
      "keywords": ["Chase", "Wells Fargo", "Citibank"],
      "action": "block",
      "tier": "deterministic"
    },
    {
      "type": "topic_boundary",
      "name": "banking_topics_only",
      "allowed_topics": ["banking", "account management", "loans"],
      "action": "flag",
      "tier": "classifier"
    },
    {
      "type": "length_limit",
      "name": "response_length",
      "max_words": 200,
      "action": "block",
      "tier": "deterministic"
    },
    {
      "type": "llm_judge",
      "name": "no_investment_advice",
      "assertion": "This response does not provide specific investment advice",
      "action": "flag",
      "tier": "llm_judge"
    }
  ]
}
```

The gateway loads this file, no code changes to the application required. Every response from the bank bot goes through every contract before reaching the user.

---

**How Covenant evaluates them — the three tiers**

The core insight is that not all assertions cost the same to evaluate. You match the evaluation method to the complexity of the assertion.

**Tier 1 — Deterministic (under 1ms)**

Pure code. No model involved.
```
KeywordContract    — does the response contain forbidden words?
RegexContract      — does it match a pattern?
LengthLimitContract — is it under N words/characters?
LanguageMatchContract — is it in the right language?
SchemaContract     — is it valid JSON matching a schema?
```

These are exact, unambiguous, instant. Use them whenever the contract can be expressed as a rule.

**Tier 2 — Classifier (10-15ms)**

Semantic assertions that require understanding meaning, not just matching text.
```
TopicBoundaryContract  — is this response about banking topics?
SentimentContract      — is the sentiment within acceptable range?
```

These use `facebook/bart-large-mnli` — a zero-shot NLI (natural language inference) model. You give it the response and a hypothesis like "this text is about banking" and it returns a probability. No fine-tuning needed.

**Tier 3 — LLM Judge (100-300ms)**

Complex nuanced assertions that require genuine reasoning.
```
LLMJudgeContract — does this response constitute investment advice?
```

A small judge LLM evaluates the assertion in natural language. "Does the following response provide specific investment advice? Respond yes or no." This is the only tier that can handle genuinely ambiguous compliance questions — the difference between explaining what a savings account is versus recommending a specific fund.

**The execution model — block vs flag**

This is where Covenant's architecture is smart about latency.
```
BLOCK contracts  → run synchronously, in parallel
                 → user waits for them
                 → only deterministic + classifier (fast)

FLAG contracts   → fire as background asyncio.Task
                 → user never waits
                 → LLM judge always goes here
```

So the user experiences: LLM call (500ms) + fast contract checks (10ms) = 510ms total. The expensive LLM judge runs in the background, logs the result to Langfuse, triggers a drift alert if needed. Zero added latency for complex evaluation.

**Drift detection**

Every contract evaluation produces a compliance score between 0.0 and 1.0 — not just pass/fail. These scores go into Redis time series per `(app_id, contract_id)`.
```
bank-support-bot / banking_topics_only / 2026-02-19 → avg 0.97
bank-support-bot / banking_topics_only / 2026-02-20 → avg 0.94  
bank-support-bot / banking_topics_only / 2026-02-21 → avg 0.87 ← ALERT
```

When compliance drops more than 10% relative to the 7-day rolling baseline, Covenant fires a `DriftAlert`. Something changed — model update, prompt exploitation, shift in user behavior. You find out before your compliance officer does.

---

**How everyone else does it — the landscape**

**Guardrails AI** — open source library, $7.5M seed. You import it into your application code and wrap your LLM calls. Works well but requires modifying every application. Not a gateway layer. No drift detection. Not self-hostable as infrastructure.

**NeMo Guardrails** — NVIDIA's framework. Programmable rails using a custom language called Colang. Powerful but complex, requires deep integration into application code, steep learning curve.

**LangChain/LlamaIndex output parsers** — schema validation and retry logic baked into the framework. Very basic, handles structured output validation only, not behavioral compliance.

**Lakera Guard** — SaaS API you call inline. Prompt injection detection and content filtering. No behavioral contracts, no self-hostable option, data leaves your infrastructure.

**Portkey guardrails** — part of their gateway, SaaS, basic content filtering and PII detection. No contract DSL, no drift detection.

**Azure Content Safety / OpenAI moderation** — provider-side content filtering. You have no control over the rules, no visibility into how they work, and your data goes to their infrastructure.

---

**The actual gap Covenant fills**

Put it in a table:
```
                    Gateway  Self-host  Contract DSL  Drift detection
Guardrails AI         ✗         ✓           partial        ✗
NeMo Guardrails       ✗         ✓           partial        ✗
Lakera Guard          ✓         ✗              ✗            ✗
Portkey               ✓         ✗           basic          ✗
Covenant              ✓         ✓             ✓             ✓