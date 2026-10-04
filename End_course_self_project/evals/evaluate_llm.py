"""LLM analysis evaluation with offline dry-run (default) or optional ``--live``.

Default mode uses a FakeClient / schema check and does NOT call OpenRouter.
Run::

    python -m evals.evaluate_llm
    python -m evals.evaluate_llm --live --limit 9
    python -m evals.evaluate_llm --live --limit 9 --full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from main.baselines import get_retriever
from main.config import PROJECT_ROOT, load_config
from main.observability import EventLogger, estimate_cost
from main.pipeline import analyze_message
from main.retrieve import load_quotes
from main.schemas import AnalysisResult, parse_analysis_json

CASES_PATH = PROJECT_ROOT / "tests" / "cases.json"
RESULTS_PATH = PROJECT_ROOT / "evals" / "results_llm.json"
DEFAULT_SUBSET = 12

RESULT_TYPE_ORDER = (
    "exact",
    "normalized",
    "variation",
    "paraphrase",
    "embedded",
    "multi",
    "cross_source",
    "no_hit",
    "near_miss",
)

ProgressCallback = Callable[[str, dict[str, Any] | None], None]


class FakeClient:
    """Deterministic stub that returns schema-valid JSON from retrieval context."""

    def __init__(self, *, echo_safety: list[str] | None = None) -> None:
        self.echo_safety = list(echo_safety or [])
        self.calls: list[dict[str, Any]] = []
        self._tokens = {"in": 120, "out": 80}
        self.model = "fake/eval-stub"

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
        )
        return self._build_reply(prompt)

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        return self.generate(
            prompt,
            system=system,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    @property
    def tokens(self) -> dict[str, int]:
        return dict(self._tokens)

    def _build_reply(self, prompt: str) -> str:
        has_notes = "Retrieved lines: none." not in prompt
        matched: list[dict[str, Any]] = []
        if has_notes:
            matched.append(
                {
                    "source": "stub",
                    "quote": "stub quote",
                    "matched_text": "stub quote",
                    "score": 1.0,
                    "meaning": "stub meaning",
                    "chat_usage": "stub usage",
                }
            )
        payload = {
            "user_text": "stub",
            "intent": "eval_stub",
            "used_knowledge_base": has_notes,
            "matched_quotes": matched,
            "interpretation": "Offline stub interpretation for schema validation.",
            "confidence": 0.5 if has_notes else 0.4,
            "uncertainty": "stub uncertainty",
            "safety_flags": list(self.echo_safety),
            "evidence": ["offline stub"],
        }
        return json.dumps(payload)


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads((path or CASES_PATH).read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("cases.json must contain a list under 'cases'")
    return cases


def stratified_subset(cases: list[dict[str, Any]], limit: int = DEFAULT_SUBSET) -> list[dict[str, Any]]:
    """Pick a stratified sample across result types (round-robin by type order)."""
    by_type: dict[str, list[dict[str, Any]]] = {rt: [] for rt in RESULT_TYPE_ORDER}
    for case in cases:
        rt = str(case["result_type"])
        by_type.setdefault(rt, []).append(case)

    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < limit:
        progressed = False
        for rt in RESULT_TYPE_ORDER:
            bucket = by_type.get(rt) or []
            if index < len(bucket):
                selected.append(bucket[index])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        index += 1
    return selected


def _api_key_available() -> bool:
    return bool(os.environ.get("MY_PRIVATE_OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY"))


def _ensure_api_key_for_live() -> str | None:
    """Return an API key for live mode, prompting interactively if needed."""
    if _api_key_available():
        return os.environ.get("MY_PRIVATE_OPENROUTER_KEY") or os.environ.get(
            "OPENROUTER_API_KEY"
        )

    import getpass

    if not sys.stdin.isatty():
        return None

    try:
        pasted = getpass.getpass("Paste your OpenRouter API key (hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not pasted:
        return None
    os.environ["OPENROUTER_API_KEY"] = pasted
    return pasted


def _schema_valid(raw: str) -> tuple[bool, AnalysisResult | None, str | None]:
    try:
        parsed = parse_analysis_json(raw)
        return True, parsed, None
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"


def _clip(text: str, width: int = 72) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= width:
        return cleaned
    return cleaned[: width - 3] + "..."


def _print_bullet_list(title: str, items: list[Any]) -> None:
    print(f"  {title}:", flush=True)
    if not items:
        print("    (none)", flush=True)
        return
    for item in items:
        print(f"    - {item}", flush=True)


def make_progress_reporter(*, full: bool = False) -> ProgressCallback:
    """Build a console progress callback.

    ``full=False``: compact status + short interpretation preview.
    ``full=True``: complete case input and structured/raw model output.
    """

    def report(event: str, payload: dict[str, Any] | None = None) -> None:
        data = payload or {}
        if event == "start":
            mode = data.get("mode", "?")
            model = data.get("model", "?")
            n = data.get("n_cases", 0)
            detail = "full" if full else "compact"
            print(
                f"LLM eval start: mode={mode} model={model} n={n} display={detail}",
                flush=True,
            )
            if full:
                print(
                    f"max_new_tokens={data.get('max_new_tokens')} "
                    f"retrieval={data.get('retrieval_method')}",
                    flush=True,
                )
            print("-" * 72, flush=True)
            return

        if event == "case_start":
            i = data["index"]
            n = data["n_cases"]
            if full:
                print(f"[{i}/{n}] {data['id']} ({data['result_type']})", flush=True)
                print("  input text:", flush=True)
                print(f"    {data.get('text', '')}", flush=True)
                _print_bullet_list(
                    "expected_quotes",
                    list(data.get("expected_quotes") or []),
                )
                _print_bullet_list(
                    "retrieved_quotes",
                    list(data.get("retrieved_quotes") or []),
                )
            else:
                print(
                    f"[{i}/{n}] {data['id']} ({data['result_type']}) "
                    f"text={_clip(str(data.get('text', '')), 48)}",
                    flush=True,
                )
            return

        if event == "case_done":
            ok = data.get("schema_valid") and data.get("kb_consistency")
            mark = "OK" if ok else "FAIL"
            bits = [
                f"  -> {mark}",
                f"schema={'yes' if data.get('schema_valid') else 'no'}",
                f"kb={'yes' if data.get('kb_consistency') else 'no'}",
                f"lat={data.get('latency_ms', 0):.0f}ms",
                f"tok={data.get('tokens_in', 0)}/{data.get('tokens_out', 0)}",
                f"cost=${data.get('estimated_cost_usd', 0):.6f}",
            ]
            if data.get("truncated_likely"):
                bits.append("truncated?")
            print(" ".join(bits), flush=True)

            if full:
                if data.get("parse_error"):
                    print(f"  parse_error:\n    {data['parse_error']}", flush=True)
                structured = data.get("structured_result")
                if structured is not None:
                    print("  structured_result:", flush=True)
                    print(
                        json.dumps(structured, indent=2, ensure_ascii=False),
                        flush=True,
                    )
                raw = data.get("raw_reply")
                if raw:
                    print("  raw_reply:", flush=True)
                    print(raw, flush=True)
                print("-" * 72, flush=True)
            else:
                if data.get("parse_error"):
                    print(
                        f"     parse_error: {_clip(str(data['parse_error']), 100)}",
                        flush=True,
                    )
                if data.get("interpretation_preview"):
                    print(
                        "     interpretation: "
                        f"{_clip(str(data['interpretation_preview']), 100)}",
                        flush=True,
                    )
            return

        if event == "summary":
            if not full:
                print("-" * 72, flush=True)
            s = data
            print(
                f"Done: n={s.get('n_cases')} "
                f"schema_validity={s.get('schema_validity_rate', 0):.4f} "
                f"kb_consistency={s.get('kb_consistency_rate', 0):.4f}",
                flush=True,
            )
            if s.get("mean_latency_ms") is not None:
                print(
                    f"mean_latency_ms={s['mean_latency_ms']:.1f} "
                    f"tokens in={s.get('total_tokens_in')} out={s.get('total_tokens_out')} "
                    f"cost_usd={s.get('total_estimated_cost_usd', 0):.6f}",
                    flush=True,
                )
            failures = s.get("failures") or []
            if failures:
                print(f"failures ({len(failures)}):", flush=True)
                for row in failures:
                    reason = row.get("parse_error") or "kb_consistency mismatch"
                    if full:
                        print(
                            f"  - {row['id']} ({row['result_type']}): {reason}",
                            flush=True,
                        )
                    else:
                        print(
                            f"  - {row['id']} ({row['result_type']}): "
                            f"{_clip(str(reason), 90)}",
                            flush=True,
                        )
            else:
                print("failures: none", flush=True)
            return

    return report


def evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    live: bool,
    log_path: Path | None = None,
    progress: ProgressCallback | None = None,
    quiet: bool = False,
    full: bool = False,
) -> dict[str, Any]:
    cfg = load_config()
    quotes = load_quotes(cfg.knowledge_path())
    logger = EventLogger(path=log_path) if log_path else EventLogger()
    if progress is not None:
        report = progress
    elif quiet:
        report = None
    else:
        report = make_progress_reporter(full=full)

    mode = "live" if live else "dry_run"
    if live and not _ensure_api_key_for_live():
        return {
            "mode": "skipped",
            "reason": (
                "no OpenRouter API key available; set OPENROUTER_API_KEY "
                "(or MY_PRIVATE_OPENROUTER_KEY), or paste it when prompted"
            ),
            "n_cases": 0,
            "summary": {},
            "cases": [],
        }

    client: Any
    if live:
        from main.API_model import OpenRouterClient

        client = OpenRouterClient(
            model=cfg.model,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
        )
    else:
        client = FakeClient()

    if report is not None:
        report(
            "start",
            {
                "mode": mode,
                "model": str(getattr(client, "model", cfg.model)),
                "n_cases": len(cases),
                "max_new_tokens": cfg.max_new_tokens,
                "retrieval_method": cfg.retrieval.method,
            },
        )

    rows: list[dict[str, Any]] = []
    schema_ok = 0
    kb_consistent = 0
    latencies: list[float] = []
    tokens_in_total = 0
    tokens_out_total = 0
    cost_total = 0.0
    failures: list[dict[str, Any]] = []

    retriever = get_retriever(cfg.retrieval.method, quotes=quotes)
    token_budget = cfg.max_new_tokens

    for index, case in enumerate(cases, start=1):
        expected = list(case.get("expected_quotes") or [])
        retrieved = retriever.search(case["text"], limit=cfg.retrieval.limit)
        retrieved_quotes = [m.entry.quote for m in retrieved]
        expect_kb = bool(retrieved_quotes)

        if report is not None:
            report(
                "case_start",
                {
                    "index": index,
                    "n_cases": len(cases),
                    "id": case["id"],
                    "result_type": case["result_type"],
                    "text": case.get("text", ""),
                    "expected_quotes": expected,
                    "retrieved_quotes": retrieved_quotes,
                },
            )

        before_in = int(client.tokens.get("in", 0))
        before_out = int(client.tokens.get("out", 0))
        started = time.perf_counter()
        analysis = analyze_message(
            case["text"],
            client,
            config=cfg,
            logger=logger,
            mode=f"eval_llm_{mode}",
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        after_in = int(client.tokens.get("in", 0))
        after_out = int(client.tokens.get("out", 0))
        delta_in = after_in - before_in
        delta_out = after_out - before_out

        raw = analysis.raw_reply or analysis.analysis
        valid, parsed, err = _schema_valid(raw)
        if valid:
            schema_ok += 1

        used_kb = parsed.used_knowledge_base if parsed is not None else None
        consistent = used_kb is not None and used_kb == expect_kb
        if consistent:
            kb_consistent += 1

        truncated_likely = live and delta_out >= token_budget
        cost = estimate_cost(
            str(getattr(client, "model", cfg.model)),
            delta_in,
            delta_out,
            cfg.pricing_per_1m_tokens,
        )
        latencies.append(latency_ms)
        tokens_in_total += delta_in
        tokens_out_total += delta_out
        cost_total += cost

        structured = parsed.model_dump() if parsed is not None else None
        interpretation_preview = ""
        if parsed is not None:
            interpretation_preview = parsed.interpretation
        elif raw:
            interpretation_preview = raw

        row: dict[str, Any] = {
            "id": case["id"],
            "result_type": case["result_type"],
            "text": case.get("text", ""),
            "expected_quotes": expected,
            "retrieved_quotes": retrieved_quotes,
            "schema_valid": valid,
            "parse_error": err,
            "used_knowledge_base": used_kb,
            "kb_consistency": consistent,
            "truncated_likely": truncated_likely,
            "latency_ms": latency_ms,
            "tokens_in": delta_in,
            "tokens_out": delta_out,
            "estimated_cost_usd": cost,
            "structured_result": structured,
            "raw_reply": raw,
        }
        rows.append(row)

        if not valid or not consistent:
            failures.append(
                {
                    "id": case["id"],
                    "result_type": case["result_type"],
                    "parse_error": err,
                    "used_knowledge_base": used_kb,
                    "expect_kb": expect_kb,
                    "truncated_likely": truncated_likely,
                }
            )

        if report is not None:
            report(
                "case_done",
                {
                    **row,
                    "interpretation_preview": interpretation_preview,
                },
            )

    n = len(rows)
    summary: dict[str, Any] = {
        "mode": mode,
        "n_cases": n,
        "schema_validity_rate": (schema_ok / n) if n else 0.0,
        "kb_consistency_rate": (kb_consistent / n) if n else 0.0,
        "failures": failures,
    }
    if live:
        summary.update(
            {
                "mean_latency_ms": sum(latencies) / n if n else 0.0,
                "total_tokens_in": tokens_in_total,
                "total_tokens_out": tokens_out_total,
                "total_estimated_cost_usd": cost_total,
                "max_new_tokens": token_budget,
            }
        )
    else:
        summary["note"] = (
            "dry-run / schema-check with FakeClient; no OpenRouter call. "
            "Use --live to measure latency/tokens/cost with a real client."
        )

    if report is not None:
        report("summary", summary)

    return {"mode": mode, "summary": summary, "cases": rows}


def run_evaluation(
    *,
    live: bool = False,
    limit: int = DEFAULT_SUBSET,
    cases_path: Path | None = None,
    out_path: Path | None = None,
    quiet: bool = False,
    full: bool = False,
) -> dict[str, Any]:
    cases = stratified_subset(load_cases(cases_path), limit=limit)
    result = evaluate_cases(cases, live=live, quiet=quiet, full=full)
    payload = {
        "cases_path": str((cases_path or CASES_PATH).resolve()),
        "subset_limit": limit,
        "live_requested": live,
        "display": "full" if full else ("quiet" if quiet else "compact"),
        **result,
    }
    destination = out_path or RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM evaluation (dry-run schema check by default; --live for API).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call OpenRouter. Requires an API key. Default is FakeClient dry-run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SUBSET,
        help=f"Stratified case subset size (default: {DEFAULT_SUBSET}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_PATH,
        help=f"Output JSON path (default: {RESULTS_PATH}).",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=CASES_PATH,
        help=f"Cases JSON path (default: {CASES_PATH}).",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case progress (still prints final path/summary).",
    )
    display.add_argument(
        "--full",
        action="store_true",
        help="Print full case input and full structured/raw model output for each case.",
    )
    args = parser.parse_args(argv)

    payload = run_evaluation(
        live=args.live,
        limit=args.limit,
        cases_path=args.cases,
        out_path=args.out,
        quiet=args.quiet,
        full=args.full,
    )
    print(f"Wrote {args.out}")
    if payload.get("mode") == "skipped":
        print("mode=skipped")
        print(payload.get("reason"))
        return 0
    if args.quiet:
        summary = payload.get("summary") or {}
        print(f"mode={payload.get('mode')}")
        print(f"n={summary.get('n_cases')}")
        print(f"schema_validity_rate={summary.get('schema_validity_rate', 0):.4f}")
        print(f"kb_consistency_rate={summary.get('kb_consistency_rate', 0):.4f}")
        if payload.get("mode") == "live":
            print(f"mean_latency_ms={summary.get('mean_latency_ms', 0):.1f}")
            print(
                f"tokens in={summary.get('total_tokens_in')} "
                f"out={summary.get('total_tokens_out')} "
                f"cost_usd={summary.get('total_estimated_cost_usd', 0):.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
