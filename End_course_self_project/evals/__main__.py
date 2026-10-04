"""Package entry: ``python -m evals`` prints available eval modules."""

from __future__ import annotations


def main() -> int:
    print("Available eval modules:")
    print("  python -m evals.evaluate_retrieval [--method fuzzy]")
    print("  python -m evals.evaluate_llm [--live] [--limit N]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
