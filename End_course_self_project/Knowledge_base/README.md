# Knowledge base (data explainer)

This folder is the **curated quote data** used by retrieval and by the LLM as grounded notes.

## Layout

```text
Knowledge_base/
  Harry_Potter/quotes.json
  The_Lord_Of_The_Rings/quotes.json
  Rick_And_Morty/quotes.json
```

Each `quotes.json` is a JSON **array** of objects.

## Record schema

| Field | Type | Role |
|-------|------|------|
| `quote` | string | Canonical line used for matching and display |
| `meaning` | string | Short English gloss (passed to the LLM, not primary match key) |
| `chat_usage` | string | How the line typically appears in chat (LLM grounding) |
| `variations` | string[] | Memetic / paraphrased forms that expand retrieval coverage |

Franchise identity comes from the **folder name** / loaded `source` label (not a separate JSON field in the file).

## Scale and provenance

- **~72 quotes** total across three franchises (~24 each).
- Hand-curated for the course project (not a web crawl dump).
- Some popular lines may be imperfectly attributed in fan culture; they are **kept** for chat-recognition utility. See [`../docs/responsible_ai.md`](../docs/responsible_ai.md).

## How code uses this data

1. `main/retrieve.py` → `load_quotes()` walks `*/quotes.json`.
2. Retrievers score the user message against `quote` + `variations`.
3. On hits, `meaning` / `chat_usage` are attached to the LLM prompt.
4. Offline tests and evals treat these files as the ground-truth catalog.

## Editing guidance

- Prefer adding a `variation` over inventing a new near-duplicate quote.
- Keep `meaning` / `chat_usage` concrete and short.
- After edits, regenerate or re-check cases: `python -m tests.build_cases` (if templates depend on quotes) and `python -m evals.evaluate_retrieval --all`.
