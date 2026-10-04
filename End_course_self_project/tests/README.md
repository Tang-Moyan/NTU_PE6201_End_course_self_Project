# Tests & labeled cases (data/evals explainer)

## Packages

| Path | Role |
|------|------|
| `cases.json` | 216 labeled chat messages for retrieval (+ LLM eval sampling) |
| `build_cases.py` | Regenerates `cases.json` from KB-aware templates |
| `rai_cases.json` | ~18 Responsible AI adversarial messages |
| `test_flow.py` | Retrieval catalog + pipeline FakeClient wiring |
| `test_baselines.py` | TF-IDF / embedding / classifier smoke |
| `test_agent.py` | Bounded agent loop + fallback |
| `test_rai.py` | RAI expectations (retrieval + stubbed analysis) |

## `cases.json` structure

Top-level: `{ "cases": [ ... ] }`.

Each case includes roughly:

- `id` — e.g. `exact-001`
- `result_type` — one of nine types (24 each → 216 total)
- `text` — simulated chat message
- `expected_quotes` — canonical quote strings expected from retrieval (empty for `no_hit` / `near_miss`)

**Result types:** `exact`, `normalized`, `variation`, `paraphrase`, `embedded`, `multi`, `cross_source`, `no_hit`, `near_miss`.

## How to run

```bash
python -m unittest discover -s tests -v
python -m tests.build_cases          # regenerate cases.json
python -m unittest tests.test_rai -v
```

Tests use a **FakeClient** (no API key). Live LLM scoring is under `evals/`, not unittest.
