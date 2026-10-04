# Product & course strategy

## Problem

Online chat is dense with franchise memes and half-remembered quotes. Readers often miss tone (comfort, threat, irony) or assume the wrong source. A lightweight **quote-aware interpreter** can:

- surface the likely source and meaning,
- explain how the line is used in chat,
- flag low-confidence or risky attributions,
- do so without requiring a full fan wiki UI.

## Target users (course persona)

1. **Course demonstrator** — show retrieve→LLM grounding, evals, and RAI in one repo (see [demo video](https://youtu.be/kfWDFcs7jhk)).
2. **Community moderator / fandom literacy helper** — quick read of “what is this line doing in this message?”
3. **Developer learning PE6201** — reference implementation spanning six course blocks.

Not targeting: real-time game chat at scale, legal rights clearance, or consumer mobile apps.

## Value proposition

| Layer | Value |
|-------|--------|
| Curated KB | Controllable evidence; educational glosses |
| Offline retrieval | Cheap, inspectable, no embedding API dependency |
| LLM analysis | Natural explanation + structured fields for metrics |
| Observability | Cost / latency / error trail for product ops |
| RAI suite | Explicit failure modes before “ship” narrative |

## Build vs buy

| Capability | Choice | Rationale |
|------------|--------|-----------|
| Quote KB | **Build** | Small, curated, course-owned meanings |
| Fuzzy / TF-IDF retrieval | **Build** (open libs) | Transparent scores; offline eval |
| LLM inference | **Buy** (OpenRouter) | Avoid hosting GPUs; swap models via config |
| Embeddings API | **Not bought** | Local hashing baseline only — cost & privacy |
| Frontend / OCR | **Not built** | Out of scope; same core can later expose HTTP |
| Vector DB / RAG platform | **Not bought** | 72 quotes fit in memory; platform overhead unjustified |

## Scope

### In scope

- Three franchises, curated JSON KB
- Fuzzy default retrieval + classical ML baselines
- Structured GenAI analysis via OpenRouter
- Bounded tool-using agent
- Offline evals, unit tests, JSONL logging
- Responsible AI adversarial cases + policy doc

### Out of scope

- Product UI / mobile app
- Image or screenshot OCR for quotes
- Physical AI / robotics
- Continuous crawl of the open web for new quotes
- Fine-tuning a custom LLM
- Production multi-tenant auth / billing

## Success metrics (product + course)

| Metric | How measured | Target |
|--------|--------------|-----------------|
| Retrieval P@1 on labeled positives | `evals/evaluate_retrieval.py` | ≥ 0.95 (fuzzy meets 1.0) |
| False positive rate on near-miss / no-hit | same | ≤ 0.05 (fuzzy meets 0.0) |
| Structured JSON parse rate (FakeClient / live) | `evals/evaluate_llm.py` | high on scripted clients |
| Latency logged | `logs/*.jsonl` | recorded per request |
| Cost estimate | same | recorded; pricing table in config |
| RAI case pass | `tests/test_rai.py` | all cases green |
| Six-block narrative | this doc + README table | complete |

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Misattributed popular quotes | Keep for chat utility; document + `misattribution_risk` flag |
| LLM hallucination of evidence | Pass only retrieved records; require `evidence` ids |
| Over-retrieval (TF-IDF/embedding) | Default to fuzzy; publish FPR comparison |
| API cost / key leakage | Env-based secrets; FakeClient in CI; `.gitignore` for `.env` / logs |
| Agent runaway loops | Max steps + tool whitelist + pipeline fallback |

## Roadmap (optional post-course)

1. HTTP API wrapping `analyze_message`
2. Human review UI for KB corrections (still no OCR)
3. Calibrated confidence from retrieval score + model self-report
4. Optional commercial embedding backend behind the same retriever interface

## Mapping to six PE6201 blocks

1. **Strategy** — this document (problem, build/buy, metrics, scope).
2. **GenAI systems** — prompts, structured output, OpenRouter client, grounding.
3. **Classical ML / IR** — fuzzy, TF-IDF, hashing embeddings, source classifier, retrieval eval.
4. **Agentic AI** — bounded ReAct tools in `main/agent.py`.
5. **Evals & observability** — unittest catalog, eval harnesses, EventLogger.
6. **Responsible AI** — `docs/responsible_ai.md` + adversarial tests.
