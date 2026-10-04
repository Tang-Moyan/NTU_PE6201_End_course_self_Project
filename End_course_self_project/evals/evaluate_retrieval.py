"""Offline retrieval evaluation over ``tests/cases.json``.

No OpenRouter / LLM calls. Run::

    python -m evals.evaluate_retrieval [--method fuzzy]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from main.baselines import get_retriever
from main.config import PROJECT_ROOT, load_config
from main.retrieve import QuoteEntry, load_quotes

CASES_PATH = PROJECT_ROOT / "tests" / "cases.json"
RESULTS_PATH = PROJECT_ROOT / "evals" / "results_retrieval.json"

MULTI_TYPES = frozenset({"multi", "cross_source"})
NEGATIVE_TYPES = frozenset({"no_hit", "near_miss"})
KNOWN_METHODS = ("fuzzy", "tfidf", "embedding")


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _precision_at_1(
    retrieved: list[str],
    expected: list[str],
    result_type: str,
) -> float:
    """Single-hit: top-1 exact match. Multi/cross: Jaccard of retrieved vs expected sets."""
    if result_type in MULTI_TYPES:
        return _jaccard(set(retrieved), set(expected))
    if result_type in NEGATIVE_TYPES:
        return 1.0 if not retrieved else 0.0
    if not expected:
        return 1.0 if not retrieved else 0.0
    if not retrieved:
        return 0.0
    return 1.0 if retrieved[0] == expected[0] else 0.0


def _recall(retrieved: list[str], expected: list[str]) -> float | None:
    if not expected:
        return None
    found = set(retrieved)
    return sum(1 for quote in expected if quote in found) / len(expected)


def _set_equality(retrieved: list[str], expected: list[str]) -> float:
    return 1.0 if set(retrieved) == set(expected) else 0.0


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads((path or CASES_PATH).read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("cases.json must contain a list under 'cases'")
    return cases


def available_methods() -> list[str]:
    methods: list[str] = []
    quotes = load_quotes()
    for name in KNOWN_METHODS:
        try:
            get_retriever(name, quotes=quotes)
            methods.append(name)
        except Exception:
            continue
    return methods or ["fuzzy"]


def evaluate_method(
    cases: list[dict[str, Any]],
    quotes: tuple[QuoteEntry, ...],
    *,
    method: str,
    limit: int,
) -> dict[str, Any]:
    retriever = get_retriever(method, quotes=quotes)
    per_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for case in cases:
        result_type = str(case["result_type"])
        expected = list(case.get("expected_quotes") or [])
        matches = retriever.search(case["text"], limit=limit)
        retrieved = [match.entry.quote for match in matches]

        p_at_1 = _precision_at_1(retrieved, expected, result_type)
        recall = _recall(retrieved, expected)
        set_eq = _set_equality(retrieved, expected)
        jaccard = _jaccard(set(retrieved), set(expected))
        false_positive = bool(retrieved) if result_type in NEGATIVE_TYPES else None

        row = {
            "id": case["id"],
            "result_type": result_type,
            "expected_quotes": expected,
            "retrieved_quotes": retrieved,
            "precision_at_1": p_at_1,
            "recall": recall,
            "set_equality": set_eq,
            "jaccard": jaccard,
            "false_positive": false_positive,
        }
        rows.append(row)
        per_type[result_type].append(row)

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    type_breakdown: dict[str, Any] = {}
    for result_type, type_rows in sorted(per_type.items()):
        recalls = [r["recall"] for r in type_rows if r["recall"] is not None]
        fps = [r["false_positive"] for r in type_rows if r["false_positive"] is not None]
        type_breakdown[result_type] = {
            "n": len(type_rows),
            "precision_at_1": _mean([float(r["precision_at_1"]) for r in type_rows]),
            "recall": _mean([float(v) for v in recalls]) if recalls else None,
            "set_equality": _mean([float(r["set_equality"]) for r in type_rows]),
            "jaccard": _mean([float(r["jaccard"]) for r in type_rows]),
            "false_positive_rate": (
                sum(1 for v in fps if v) / len(fps) if fps else None
            ),
        }

    positive_rows = [r for r in rows if r["result_type"] not in NEGATIVE_TYPES]
    negative_rows = [r for r in rows if r["result_type"] in NEGATIVE_TYPES]
    recalls_all = [r["recall"] for r in positive_rows if r["recall"] is not None]

    summary = {
        "method": method,
        "n_cases": len(rows),
        "precision_at_1": _mean([float(r["precision_at_1"]) for r in rows]),
        "recall": _mean([float(v) for v in recalls_all]) if recalls_all else None,
        "set_equality": _mean([float(r["set_equality"]) for r in rows]),
        "jaccard": _mean([float(r["jaccard"]) for r in rows]),
        "false_positive_rate": (
            sum(1 for r in negative_rows if r["false_positive"]) / len(negative_rows)
            if negative_rows
            else None
        ),
        "n_negative": len(negative_rows),
        "by_result_type": type_breakdown,
    }
    return {"summary": summary, "cases": rows}


def run_evaluation(
    *,
    method: str = "fuzzy",
    cases_path: Path | None = None,
    out_path: Path | None = None,
    limit: int | None = None,
    all_available: bool = False,
) -> dict[str, Any]:
    cfg = load_config()
    cases = load_cases(cases_path)
    quotes = load_quotes(cfg.knowledge_path())
    top_k = cfg.retrieval.limit if limit is None else limit

    methods_to_run = available_methods() if all_available else [method.strip().lower()]
    methods_payload: dict[str, Any] = {}
    for name in methods_to_run:
        methods_payload[name] = evaluate_method(
            cases,
            quotes,
            method=name,
            limit=top_k,
        )

    primary = method.strip().lower()
    if primary not in methods_payload:
        methods_payload[primary] = evaluate_method(
            cases,
            quotes,
            method=primary,
            limit=top_k,
        )

    payload = {
        "cases_path": str((cases_path or CASES_PATH).resolve()),
        "available_methods": available_methods(),
        "limit": top_k,
        "methods": methods_payload,
        "primary_method": primary,
        "summary": methods_payload[primary]["summary"],
    }

    destination = out_path or RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline retrieval evaluation (no LLM).")
    parser.add_argument(
        "--method",
        default="fuzzy",
        help="Retrieval method: fuzzy (default), tfidf, or embedding.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also evaluate every importable baseline method.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max matches per query (defaults to config retrieval.limit).",
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
    args = parser.parse_args(argv)

    payload = run_evaluation(
        method=args.method,
        cases_path=args.cases,
        out_path=args.out,
        limit=args.limit,
        all_available=args.all,
    )
    print(f"Wrote {args.out}")

    if args.all and len(payload["methods"]) > 1:
        print("comparison:")
        for name, block in payload["methods"].items():
            s = block["summary"]
            recall = f"{s['recall']:.4f}" if s["recall"] is not None else "n/a"
            fpr = (
                f"{s['false_positive_rate']:.4f}"
                if s["false_positive_rate"] is not None
                else "n/a"
            )
            print(
                f"  {name}: n={s['n_cases']} p@1={s['precision_at_1']:.4f} "
                f"recall={recall} fpr={fpr}"
            )
        print()

    summary = payload["summary"]
    print(f"primary method={summary['method']} n={summary['n_cases']}")
    print(f"precision@1={summary['precision_at_1']:.4f}")
    print(f"recall={summary['recall']:.4f}" if summary["recall"] is not None else "recall=n/a")
    print(
        f"false_positive_rate={summary['false_positive_rate']:.4f}"
        if summary["false_positive_rate"] is not None
        else "false_positive_rate=n/a"
    )
    print("by_result_type:")
    for result_type, stats in summary["by_result_type"].items():
        p = stats["precision_at_1"]
        r = stats["recall"]
        fpr = stats["false_positive_rate"]
        recall_s = f"{r:.4f}" if r is not None else "n/a"
        fpr_s = f"{fpr:.4f}" if fpr is not None else "n/a"
        print(f"  {result_type}: n={stats['n']} p@1={p:.4f} recall={recall_s} fpr={fpr_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
