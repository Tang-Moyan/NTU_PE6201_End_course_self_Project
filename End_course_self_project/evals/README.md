# Evaluations (evals explainer)

This package holds **offline and live evaluation harnesses** plus result JSON.

## What is here

| Path | Purpose |
|------|---------|
| `evaluate_retrieval.py` | No-LLM retrieval metrics on `tests/cases.json` |
| `evaluate_llm.py` | Dry-run (FakeClient) or `--live` OpenRouter analysis metrics |
| `results_retrieval.json` | Latest multi-method retrieval scores |
| `results_llm.json` | Latest LLM eval output (includes full live n=216 run when regenerated) |
| `__main__.py` | `python -m evals` helper hints |

## Retrieval eval

```bash
python -m evals.evaluate_retrieval --all
# or: --method fuzzy|tfidf|embedding
```

**Metrics:** precision@1, recall (positives), false-positive rate (negatives / near-miss / no-hit), breakdown by `result_type`.

**Latest headline (fuzzy default):** P@1 = 1.0, recall = 1.0, FPR = 0.0 on 216 cases. TF-IDF / embedding baselines are worse on FPR (see JSON / README table).

## LLM eval

```bash
python -m evals.evaluate_llm                  # FakeClient dry-run
python -m evals.evaluate_llm --live --limit 9 # cheap smoke
python -m evals.evaluate_llm --live --limit 216
python -m evals.evaluate_llm --live --limit 9 --full  # verbose printout
```

**Metrics:** schema validity rate, KB consistency (`used_knowledge_base` vs retrieval emptiness), latency, tokens, estimated USD cost.

**Latest full live run:** n=216, schema ≈ 0.981, KB consistency ≈ 0.981, mean latency ≈ 7.3s, cost ≈ $0.058. Four failures were malformed JSON (not token truncation).

## Design notes / limits

- Retrieval eval is strong and deterministic.
- `kb_consistency` does **not** grade interpretation quality—only the KB-usage flag vs hits.
- Live results depend on model + prompt version; re-run to refresh `results_llm.json`.
- Cost estimates use `config.yaml` → `pricing_per_1m_tokens` (approximate).

## Related data

Labeled cases live under [`../tests/cases.json`](../tests/cases.json) (see [`../tests/README.md`](../tests/README.md)).
