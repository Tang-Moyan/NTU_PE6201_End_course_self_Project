# Quote-Aware Chat Interpretation (PE6201)

Curated quote knowledge base + offline retrieval + OpenRouter LLM analysis for short chat messages that echo lines from **Harry Potter**, **The Lord of the Rings**, and **Rick and Morty**.

This project is a course end-of-term artifact covering six PE6201 areas (see [Course coverage](#course-coverage)). Frontend UI and image/OCR quote recognition are intentionally out of scope.

## Demo video

A recorded walkthrough of the CLI demo (knowledge base → hit / miss paths → agent → retrieval evals → Responsible AI):

- [YouTube demo](https://youtu.be/kfWDFcs7jhk)

Written report: [`docs/project_report.md`](docs/project_report.md).

## Product documentation

Persona, inputs/outputs, architecture, and metrics:

- [`docs/product.md`](docs/product.md)

Data and evals:

- [`Knowledge_base/README.md`](Knowledge_base/README.md) — quote KB
- [`tests/README.md`](tests/README.md) — labeled cases + RAI fixtures
- [`evals/README.md`](evals/README.md) — retrieval / LLM harnesses + result JSON

## What it does

1. Load quote records from `Knowledge_base/*/quotes.json`.
2. Retrieve top matching records (fuzzy by default; TF-IDF / hashing embedding optional).
3. Call an LLM with the message + matched evidence.
4. Return a structured JSON analysis (`intent`, `evidence`, `confidence`, `safety_flags`, …) plus a natural-language interpretation.
5. Log latency, tokens, and estimated cost to JSONL under `logs/`.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Set OpenRouter credentials (env vars override `config.yaml`):

```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY = "sk-or-..."
# optional overrides
$env:OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

Or edit `config.yaml`:

```yaml
openrouter:
  api_key: ""          # prefer env OPENROUTER_API_KEY
  model: openai/gpt-4o-mini
retrieval:
  method: fuzzy        # fuzzy | tfidf | embedding
  limit: 5
  threshold: 72
```

## Run

Interactive CLI (pipeline mode):

```bash
python -m main
```

One-shot message:

```bash
python -m main "You shall not pass."
```

Bounded tool-using agent (ReAct-style loop):

```bash
python -m main --mode agent "Happiness can be found even in the darkest of times."
```

Offline path (no network): inject a fake client in tests; production path uses `OpenRouterClient`.

## Project layout

```text
Knowledge_base/          # Curated quotes + meaning / chat_usage / variations
main/
  retrieve.py            # Fuzzy retrieval
  baselines.py           # TF-IDF, LocalHashing embedding, source classifier
  pipeline.py            # Retrieve → LLM → structured Analysis
  agent.py               # Bounded ReAct agent (search / details / finish)
  schemas.py             # Pydantic AnalysisResult
  observability.py       # JSONL EventLogger + cost estimate
  config.py              # YAML + env config
  API_model/openrouter.py
tests/                   # Flow, retrieval catalog, baselines, agent, RAI
evals/                   # Offline retrieval / LLM harnesses
docs/                    # Architecture, strategy, RAI, project report
config.yaml
requirements.txt
```

## Tests

Unit / integration tests use a `FakeClient` (no OpenRouter calls):

```bash
python -m unittest discover -s tests -v
```

Regenerate the 216-case retrieval catalog (9 types × 24):

```bash
python -m tests.build_cases
```

Responsible AI adversarial suite:

```bash
python -m unittest tests.test_rai -v
```

## Offline evaluations

Retrieval (no LLM). Compare all importable methods:

```bash
python -m evals.evaluate_retrieval --all
# writes evals/results_retrieval.json
```

Single method:

```bash
python -m evals.evaluate_retrieval --method fuzzy
python -m evals.evaluate_retrieval --method tfidf
python -m evals.evaluate_retrieval --method embedding
```

LLM analysis metrics (FakeClient dry-run by default — **no API key needed**):

```bash
python -m evals.evaluate_llm
# writes evals/results_llm.json
```

### Optional: live OpenRouter smoke test

Unit tests (`python -m unittest …`) do **not** need a real LLM. Use a live call only when you want to check real JSON quality, latency, and cost.

**Option A — paste key when prompted (recommended for a one-off demo):**

```powershell
# from project root, venv activated
python -m evals.evaluate_llm --live --limit 9
# when asked: Paste your OpenRouter API key (hidden):
# paste key, Enter (input is hidden)

# or interactive product CLI (also prompts if no env key):
python -m main "You shall not pass."
python -m main --mode agent "Happiness can be found even in the darkest of times."
```

**Option B — set env for the current PowerShell session:**

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."   # paste your key
# optional:
# $env:OPENROUTER_MODEL = "openai/gpt-4o-mini"

python -m evals.evaluate_llm --live --limit 9
python -m main "You shall not pass."
```

`--limit 9` ≈ one case per result type (cheap smoke). Full catalog:

```powershell
python -m evals.evaluate_llm --live --limit 216
```

Latest full live run (n=216): schema validity ≈ 0.981, KB consistency ≈ 0.981, mean latency ≈ 7.3s, estimated cost ≈ $0.058.

Display modes:

```powershell
python -m evals.evaluate_llm --live --limit 9          # compact progress (default)
python -m evals.evaluate_llm --live --limit 9 --full   # full input + structured/raw output
python -m evals.evaluate_llm --live --limit 9 --quiet  # summary only
```

Results land in `evals/results_llm.json` (includes `text`, `structured_result`, `raw_reply`); request metrics also append under `logs/`.

If multi-quote cases fail with JSON parse errors and `tokens_out` equals `max_new_tokens`, the completion was truncated — raise `max_new_tokens` in `config.yaml` (current default: 900).

### Latest retrieval comparison (216 cases, limit=5)

| Method    | Precision@1 | Recall (positives) | False positive rate (negatives) |
|-----------|-------------|--------------------|----------------------------------|
| fuzzy     | 1.000       | 1.000              | 0.000                            |
| tfidf     | 0.864       | 1.000              | 0.604                            |
| embedding | 0.876       | 1.000              | 0.375                            |

**Default choice:** `fuzzy` — perfect P@1 / recall on this catalog and zero FPR on near-miss / no-hit cases. TF-IDF and hashing embeddings over-match on negatives; kept as baselines for the ML/IR unit.

## Course coverage

| Course block | Where it lives in this repo |
|--------------|-----------------------------|
| **1. AI Product Strategy** | `docs/strategy.md` — problem, users, build-vs-buy, scope, success metrics |
| **2. Generative AI Systems** | `main/pipeline.py`, `schemas.py`, prompts, OpenRouter client, structured JSON output |
| **3. Classical / Discriminative ML & IR** | `main/retrieve.py`, `main/baselines.py`, `evals/evaluate_retrieval.py` |
| **4. Agentic AI** | `main/agent.py` — tools `search_quotes`, `get_quote_details`, `finish_analysis`; step budget |
| **5. Evaluation & Observability** | `tests/`, `evals/`, `main/observability.py`, `logs/*.jsonl` |
| **6. Responsible AI** | `docs/responsible_ai.md`, `tests/rai_cases.json`, `tests/test_rai.py` |

Physical AI / robotics and a production web frontend are explicitly **out of scope**; rationale is in `docs/strategy.md`.

## Documentation

- [Product documentation](docs/product.md) — persona, I/O, architecture, metrics
- [Architecture](docs/architecture.md)
- [Product & course strategy](docs/strategy.md)
- [Responsible AI](docs/responsible_ai.md)
- [Final project report](docs/project_report.md)
- [Docs index](docs/README.md)
- [Demo video (YouTube)](https://youtu.be/kfWDFcs7jhk)

## Knowledge base note

Some popular attributions in the wild are imperfect (misquoted or misattributed lines). This project **keeps** those entries in their current franchise folders for chat-recognition utility, and documents the risk under Responsible AI rather than silently deleting them.

## License / course use

Course project for PE6201. Quote texts remain property of their respective rights holders; used here for educational analysis only.
