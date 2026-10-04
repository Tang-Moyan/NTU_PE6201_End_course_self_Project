# Responsible AI

## Purpose

This system explains chat messages that may reference franchise quotes. Mistakes can mislead users about authorship, tone, or intent. This document states risks, controls, and how we test them.

## Risk inventory

| Risk | Example | Severity |
|------|---------|----------|
| **False attribution** | Matching a near-miss line to the wrong franchise | Medium |
| **KB misattribution debt** | Popular “quote” stored under a franchise it did not originate from | Medium |
| **Ungrounded invention** | LLM invents a quote id / meaning not in evidence | High |
| **Overconfidence** | High `confidence` on ambiguous banter | Medium |
| **Jailbreak / prompt injection** | Message tries to override system instructions | High |
| **Hate / harassment adjacent** | Weaponizing a quote to insult a person | High |
| **Privacy** | Logging full messages with PII to disk | Medium |
| **Cost abuse** | Agent loop or huge prompts burning API budget | Low–Medium |

## Design controls

1. **Grounding** — The LLM receives only retrieved records. Prompts instruct it to mark `used_knowledge_base=false` when evidence is empty or irrelevant.
2. **Structured safety fields** — `safety_flags`, `uncertainty`, and `confidence` are first-class outputs for downstream filtering.
3. **Conservative retrieval default** — Fuzzy matching with a threshold; preferred over TF-IDF/embedding because of lower false-positive rate on the course catalog.
4. **Bounded agent** — Whitelisted tools, max steps, fallback to single-shot pipeline.
5. **Secrets** — API keys via environment; `.env` and `logs/` gitignored.
6. **Transparent debt** — Known imperfect attributions stay in the KB for chat-recognition utility and are called out here rather than silently deleted.

## Misattribution policy

Some lines in fan culture are widely misquoted or misattributed. For this course project:

- **Keep** entries in their current folders so retrieval still helps explain *chat usage*.
- **Do not** present the KB as a scholarly citation database.
- Prefer interpretations that say “commonly associated with …” when uncertainty is high.
- Tests and prompts encourage `misattribution_risk` (or similar) when the match is culturally loose.

## Adversarial test suite

Location: `tests/rai_cases.json` + `tests/test_rai.py`.

Case themes include:

- Jailbreak / instruction override attempts
- Requests to ignore the knowledge base or invent sources
- Near-miss lines that must not claim a confident KB hit
- Hostile or demeaning framing around a quote
- Empty / irrelevant messages (no false evidence)

Assertions (high level):

- Pipeline / FakeClient path does not claim fabricated KB ids.
- Safety-oriented cases produce expected flags or cautious uncertainty.
- Retrieval negatives stay empty at the configured threshold where required.

Run:

```bash
python -m unittest tests.test_rai -v
```

## Logging & privacy

- Event logs store operational metrics (tokens, latency, cost). Avoid committing real user logs.
- For any deployment beyond coursework, redacts PII, set retention limits, and document lawful basis for message processing.

## Human oversight

This is an **assistive** interpreter, not an automated moderator of record. For moderation or educational grading:

- A human should review low-confidence or flagged outputs.
- KB edits should go through review (wrong meaning is worse than a miss).

## Limitations (disclose to users)

- Coverage is three franchises and ~72 quotes — not the open web.
- Paraphrases outside `variations` may miss.
- LLM text can still be wrong even when JSON parses.
- Cost and latency depend on the chosen OpenRouter model.
- No image/OCR path; screenshot quotes are unsupported.

## Incident response (course-scale)

1. Disable live API key / switch CLI to FakeClient for demos if needed.
2. Reproduce with the failing message in `tests/` or `rai_cases.json`.
3. Fix prompt, threshold, or KB entry; re-run `unittest` + retrieval eval.
4. Note the change in git history / course report.
