# Quote-Aware Chat Interpretation: Final Project Report

**Course:** PE6201 · **Artifact:** curated knowledge base + retrieval + structured LLM analysis (CLI) · **Demo:** [https://youtu.be/kfWDFcs7jhk](https://youtu.be/kfWDFcs7jhk)

## 1. Problem framing and what changes versus today

Online chat is full of half-remembered franchise lines. Readers often miss tone (comfort, threat, irony) or guess the wrong source. Today’s default response is either (a) ask a general chatbot with no grounded source, or (b) search a fan wiki manually. Both are slow, inconsistently evidenced, and hard to evaluate.

This project builds a **quote-aware interpreter** for short messages that may echo lines from *Harry Potter*, *The Lord of the Rings*, or *Rick and Morty*. The system retrieves curated records first, then asks an LLM for a structured explanation (intent, evidence, confidence, safety flags, interpretation). Relative to “today,” the difference is not a consumer app with a polished UI; it is a **inspectable, testable pipeline**: evidence is printed before generation, negatives can stay unmatched, and cost/latency are logged. Frontend UI and screenshot OCR were deliberately excluded so engineering effort stayed on grounding, evaluation, and risk controls rather than presentation.

## 2. Technology choices (ILO 1)

The course asks us to distinguish emerging AI families and pick techniques for a business problem class. Mapping for this project:

| Family | Role here | Why selected / rejected |
|--------|-----------|-------------------------|
| Discriminative / classical IR | Fuzzy match (default), TF-IDF, local hashing embeddings, source classifier | Needed for cheap offline ranking and fair baselines |
| Foundation models | OpenRouter-hosted chat model (`gpt-4o-mini` default) | Natural-language interpretation + JSON structure without self-hosting GPUs |
| Retrieval-augmented systems | Retrieve-then-generate over ~72 JSON quotes | Grounding without a vector-database product |
| Agentic systems | Bounded ReAct loop (`search_quotes`, `get_quote_details`, `finish_analysis`) | Demonstrates tool use with a step budget and pipeline fallback |
| Multimodal | Out of scope | OCR of quote screenshots adds labeling cost without changing the core thesis |
| Physical AI | Out of scope | No robotics or sensor loop in the problem |

The business problem class is **assistive interpretation / fandom literacy**, not high-stakes moderation-as-a-service. That justifies a small curated KB and conservative retrieval over an always-on web RAG stack.

## 3. End-to-end solution and validation (ILO 2)

The code-based stack implements: JSON knowledge base → retriever → prompt with attached notes → structured `AnalysisResult` (Pydantic) → JSONL observability. A managed LLM platform (OpenRouter) supplies inference; configuration is YAML/env rather than a visual low-code builder. That split is intentional: for a 72-quote corpus, a drag-and-drop RAG product would add vendor lock-in and opaque scoring without improving educational transparency. “Low-code” in this project appears as **configuration and a hosted API**, while retrieval, schemas, evals, and the agent remain code-owned.

Validation layers:

1. **Unit / catalog tests** — 216 labeled chat cases across nine result types (exact, normalized, variation, paraphrase, embedded, multi, cross-source, no-hit, near-miss), plus agent and Responsible AI suites (~18 adversarial cases).
2. **Offline retrieval eval** — precision@1, recall, false-positive rate by method.
3. **Live LLM evaluation** — full labeled catalog (n=216) measuring schema validity, knowledge-base consistency, latency, tokens, and estimated USD cost (plus an earlier stratified n=9 smoke run used while tuning).

Deterministic CI uses a `FakeClient`; live calls are opt-in so routine tests do not require spending tokens.

## 4. Design alternatives, business and technical factors (ILO 3)

| Alternative | Cost / time | Risk | Accuracy / latency | Decision |
|-------------|-------------|------|--------------------|----------|
| Embeddings API + vector DB | Higher $ and ops | Privacy, lock-in | Strong semantic recall possible; overkill at n=72 | Rejected for course scale |
| TF-IDF / hashing only (no LLM) | Very low $ | Weak chat explanation | Fast; poor interpretation quality | Insufficient alone |
| LLM-only (no retrieval) | Medium $ | Hallucinated sources | Fluent but ungrounded | Rejected as primary design |
| Fuzzy + LLM (chosen) | Low retrieval $; pay-per-call LLM | Truncation / JSON errors | High labeled P@1; ~3–9s live latency | Selected |
| Open-ended agent | Higher $ and loop risk | Tool abuse, runaway steps | Flexible | Rejected; bounded agent only |

**Retrieval metrics (216 cases, top-k=5):** fuzzy achieved precision@1 = 1.0, recall = 1.0, false-positive rate = 0.0 on negatives. TF-IDF reached P@1 ≈ 0.86 with FPR ≈ 0.60; local hashing embedding P@1 ≈ 0.88 with FPR ≈ 0.38. Perfect fuzzy scores deserve critique (next section), but the **relative** ranking clearly supports the default.

**Live LLM (full catalog, after tuning):** on all 216 cases, schema validity ≈ 0.981 and KB consistency ≈ 0.981 (212/216 pass both checks); mean latency ≈ 7.3s; total tokens ≈ 122k in / 67k out; estimated cost ≈ $0.058. The four failures were malformed JSON (not token-budget truncation). An earlier n=9 smoke run looked perfect, and multi-quote truncations at `max_new_tokens=450` had already been fixed by raising the budget to 900, tightening the prompt, and requesting `json_object` response format—showing why full-catalog live eval matters after a green smoke test.

ROI for a course/demo setting remains favorable: minutes of offline retrieval eval, under roughly six cents for a full live pass, and a reusable CLI. Time-to-deploy for a real product would still require an API layer, retention policy, and human review for flagged outputs.

## 5. Performance critique, evals critique, difficulties, tuning

**Metrics critique.** Fuzzy’s perfect catalog scores are partly a success of test design (variations and paraphrases were written against the same KB) and partly a warning: they may **overstate generalization** to wild chat. Near-miss and no-hit FPR=0 is more meaningful operationally than P@1=1.0 on positives. TF-IDF/embedding’s high FPR shows why “more semantic” is not automatically better on a tiny labeled set with a thresholded product goal.

**Evals critique.** Retrieval evals are strong (full catalog, multi-method). Live LLM coverage now matches the full 216-case catalog, which is a clear improvement over the early n=9 smoke test—and it exposed a ~1.9% residual JSON parse failure rate that smoke testing missed. Still, `kb_consistency` only checks whether `used_knowledge_base` matches “any retrieval hit,” not whether the interpretation is faithful. Human rating of tone/irony is still missing. Offline `FakeClient` proves wiring and schema discipline, not model quality.

**Difficulties surpassed.** (1) Mechanical test phrases that made paraphrase/variation unrealistically easy—regenerated toward natural chat. (2) Live JSON truncation on multi / cross-source cases—diagnosed via `tokens_out == max_new_tokens`, then tuned. (3) Keeping popular misattributed lines in-folder for chat utility while documenting misattribution risk rather than silently deleting cultural noise. (4) Agent invalid tool JSON—fallback to the pipeline so demos do not die mid-loop.

**Rough edges.** No production HTTP API; confidence is model-reported, not calibrated; hashing “embeddings” are a didactic baseline, not SOTA vectors; logs may contain message text and need retention rules before any real deployment; rare malformed JSON still appears even with `json_object` mode (~4/216); live quality depends on the chosen OpenRouter model and can regress without continuous eval. A natural hardening step is retry-on-parse-failure for those residual cases.

## 6. Communication, limitations, risks (ILO 4)

For a mixed audience: the product promise is “explain what this chat line is doing, with sources you can see,” not “authoritative quotation scholarship.” The [demo video](https://youtu.be/kfWDFcs7jhk) retells that story for mixed viewers—KB fields, hit/miss CLI, bounded agent, retrieval comparison, RAI—without a product UI. Limitations: three franchises, ~72 quotes, English glosses, CLI-only, no OCR. Risks: false attribution, prompt injection, overconfidence, and privacy if logs are retained. Controls include grounded prompts, `safety_flags`, conservative fuzzy default, adversarial tests, and env-based secrets. Humans should review low-confidence or flagged outputs in moderation-like use.

## 7. Optional future path

Natural next steps: a thin HTTP API; human-in-the-loop KB correction; calibrated confidence combining retrieval score and model self-report; optional commercial embeddings behind the same retriever interface; larger wild-chat evaluation with human labels. Multimodal OCR remains optional only if screenshot quotes become a real user demand.

## Closing

The project’s value is less “another chatbot” and more a reasoned stack choice: classical retrieval where it wins on false positives, a foundation model where language explanation is needed, a bounded agent where tool use must be taught safely, and evaluation/RAI artifacts that make claims falsifiable. That combination—and the critique of where our numbers look too good—is the intended depth of the outcome.
