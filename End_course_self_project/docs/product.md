# Product documentation

High-level product spec for the quote-aware chat interpreter (PE6201).

## Persona

| Persona | Need | How this product helps |
|---------|------|------------------------|
| **Course / TA reviewer** | See grounded retrieve→LLM, evals, RAI in one repo | CLI demo + video + offline/live eval artifacts |
| **Fandom-literate reader / light moderator** | “What is this chat line doing?” | Source + meaning + chat-usage notes + structured interpretation |
| **Developer learning the stack** | Reproducible pipeline without a UI | Config-driven code path, FakeClient tests, OpenRouter live mode |

Not a consumer social app, not a scholarly citation database, not real-time game-chat infra.

## Input

| Input | Description |
|-------|-------------|
| **Primary** | A short chat message string (CLI args or interactive stdin) |
| **Optional flags** | `--mode pipeline\|agent`, `--retrieval fuzzy\|tfidf\|embedding`, `--model`, `--knowledge-dir` |
| **Config / env** | `config.yaml` + `OPENROUTER_API_KEY` (and optional model overrides) |
| **Knowledge base** | `Knowledge_base/<franchise>/quotes.json` records loaded at runtime |

Example inputs: `"You shall not pass."` (hit), `"Can you pick up milk on your way home?"` (miss).

## Output

| Output | Description |
|--------|-------------|
| **Retrieval trace** | Printed matches: source, quote, score, meaning, chat_usage (or “Matched 0”) |
| **Structured analysis** | JSON fields: `user_text`, `intent`, `used_knowledge_base`, `matched_quotes`, `interpretation`, `confidence`, `uncertainty`, `safety_flags`, `evidence` |
| **Natural-language interpretation** | Human-readable explanation (also inside structured JSON) |
| **Ops metrics** | `tokens in/out`, `latency_ms`, `request_id`; JSONL under `logs/` with estimated USD cost |

## High-level architecture

```text
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ User message│────▶│ Retriever        │────▶│ Prompt builder      │
│ (CLI input) │     │ fuzzy / tfidf /  │     │ + KB notes or none  │
└─────────────┘     │ hashing embed    │     └──────────┬──────────┘
                    └──────────────────┘                │
                              ▲                         ▼
                    ┌─────────┴────────┐     ┌─────────────────────┐
                    │ Knowledge_base/  │     │ External LLM        │
                    │ quotes.json ×3   │     │ (OpenRouter)        │
                    └──────────────────┘     └──────────┬──────────┘
                                                        │
         Agent mode (optional):                         ▼
         search_quotes / get_quote_details /   ┌─────────────────────┐
         finish_analysis (bounded steps)       │ AnalysisResult JSON │
                                               │ + EventLogger JSONL │
                                               └─────────────────────┘
```

**Code logic path:** `main/pipeline.py` (or `main/agent.py`) → `main/baselines.get_retriever` / `main/retrieve.py` → `main/API_model/openrouter.py` → `main/schemas.py` → `main/observability.py`.

## Metrics targeted vs reached

| Metric | Target | Reached |
|--------|-----------------|---------|
| Retrieval precision@1 (positives) | ≥ 0.95 | **1.0** (fuzzy, 216 cases) |
| Retrieval recall (positives) | high | **1.0** (fuzzy) |
| False-positive rate (near-miss / no-hit) | ≤ 0.05 | **0.0** (fuzzy) |
| TF-IDF / embedding baselines | documented comparison | P@1 ≈ 0.86 / 0.88; FPR ≈ 0.60 / 0.38 (not defaults) |
| Live LLM schema validity | high on catalog | **≈ 0.981** (212/216) |
| Live LLM KB-flag consistency | high on catalog | **≈ 0.981** (212/216) |
| Live mean latency | recorded | **≈ 7.3 s** |
| Live estimated cost (full catalog) | recorded | **≈ $0.058** |
| Unit tests (FakeClient) | green | **27 passed** (last full offline run) |
| RAI adversarial cases | covered + tested | **~18 cases** in `tests/rai_cases.json` |

Artifacts: `evals/results_retrieval.json`, `evals/results_llm.json`, `logs/`.

## Related docs

- Data explainer: [`../Knowledge_base/README.md`](../Knowledge_base/README.md)
- Evals explainer: [`../evals/README.md`](../evals/README.md)
- Test catalog explainer: [`../tests/README.md`](../tests/README.md)
- Architecture detail: [`architecture.md`](architecture.md)
- Final report: [`project_report.md`](project_report.md)
- Demo video: [https://youtu.be/kfWDFcs7jhk](https://youtu.be/kfWDFcs7jhk)
