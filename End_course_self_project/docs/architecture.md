# Architecture

## Goal

Given a short chat message, detect whether it echoes a curated franchise quote, attach grounded evidence from the knowledge base, and produce a structured interpretation suitable for chat moderation / fandom literacy / teaching demos.

## High-level flow

```text
User message
    │
    ▼
┌──────────────────┐
│  Retriever       │  fuzzy (default) | tfidf | embedding
│  Knowledge_base  │
└────────┬─────────┘
         │ QuoteMatch[]
         ▼
┌──────────────────┐
│  Prompt builder  │  system + user + evidence block
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLM (OpenRouter)│  JSON AnalysisResult
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  EventLogger     │  latency / tokens / cost → logs/*.jsonl
└──────────────────┘
```

**Agent mode** replaces the single retrieve→LLM call with a bounded ReAct loop:

```text
message → LLM (tool choice) ⇄ search_quotes / get_quote_details → finish_analysis
```

If the model emits invalid tool JSON, the agent falls back to the standard pipeline so the user still gets an analysis.

## Components

| Module | Responsibility |
|--------|----------------|
| `main/config.py` | Load `config.yaml` + env (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, …) |
| `main/retrieve.py` | Quote loading, fuzzy matching (`rapidfuzz`), `QuoteMatch` |
| `main/baselines.py` | `TfidfRetriever`, `LocalHashingEmbeddingRetriever`, `DiscriminativeSourceClassifier`, `get_retriever` factory |
| `main/schemas.py` | Pydantic `AnalysisResult` (intent, evidence, confidence, uncertainty, safety_flags, …) |
| `main/pipeline.py` | Orchestration, prompt, parse JSON, CLI |
| `main/agent.py` | Tool registry, step budget, fallback |
| `main/observability.py` | `EventLogger`, `estimate_cost`, request IDs |
| `main/API_model/openrouter.py` | Thin OpenAI-compatible chat client |

## Data model (knowledge base)

Each record in `Knowledge_base/<franchise>/quotes.json`:

- `id` — stable string key (used in tests and agent `get_quote_details`)
- `quote` — canonical line
- `source` — franchise / work label
- `meaning` — short English gloss
- `chat_usage` — how the line typically appears in chat
- `variations` — common paraphrases / memetic forms used for retrieval expansion

Retrieval scores against `quote` + `variations`. Meaning / chat_usage are passed to the LLM as grounded notes, not used as primary match keys.

## Structured LLM output

The model is instructed to return JSON matching `AnalysisResult`:

- `intent` — e.g. quote_reference, banter, unrelated, …
- `used_knowledge_base` — whether KB evidence was material
- `evidence` — list of quote ids / short snippets cited
- `confidence` — 0–1
- `uncertainty` — free-text caveats
- `safety_flags` — e.g. `jailbreak_attempt`, `hate_adjacent`, `misattribution_risk`
- `interpretation` — user-facing explanation

If JSON parse fails, the pipeline keeps the raw reply as `interpretation` and leaves `result=None` so observability still records the call.

## Retrieval methods

1. **Fuzzy** (`rapidfuzz` token/partial ratios + threshold) — production default; best FPR on the course catalog.
2. **TF-IDF** cosine over quote+variation documents — classical IR baseline.
3. **Local hashing embedding** — bag-of-n-grams hashed into a fixed vector; no external embedding API. Demonstrates vector retrieval without shipping a large model.

Threshold / top-k come from `config.retrieval`.

## Observability

Each analysis (pipeline or agent finish) writes one JSONL event:

- `request_id`, `model`, `prompt_version`
- `retrieval_method`, `n_matches`, `mode`
- `latency_ms`, `tokens_in`, `tokens_out`, `estimated_cost_usd`
- `error` if the call failed

Logs default under `logs/` (gitignored). Pricing table lives in config (`pricing_per_1m_tokens`) for rough USD estimates only.

## Evaluation surfaces

| Harness | Input | Output |
|---------|-------|--------|
| `tests/test_flow.py` | 216 labeled cases + FakeClient | retrieval expectations + prompt wiring |
| `tests/test_baselines.py` | smoke queries | TF-IDF / embedding / classifier |
| `tests/test_agent.py` | FakeClient tool script | agent loop + fallback |
| `tests/test_rai.py` | `rai_cases.json` | safety / refusal / no false KB claim |
| `evals/evaluate_retrieval.py` | cases.json | P@1, recall, FPR by method |
| `evals/evaluate_llm.py` | subset + FakeClient | parse rate, latency stubs, flags |

## Design choices

- **Retrieve-then-generate** keeps the LLM grounded and makes failures inspectable (CLI prints matches before the reply).
- **Structured JSON** supports product metrics (confidence histograms, safety flag rates) without scraping free text.
- **Offline FakeClient** makes CI deterministic; live OpenRouter is opt-in via env.
- **Bounded agent** demonstrates tool use without open-ended autonomy (max steps, whitelist tools, pipeline fallback).
- **No frontend / OCR** — same analysis API could later sit behind HTTP; not required for course evidence.
