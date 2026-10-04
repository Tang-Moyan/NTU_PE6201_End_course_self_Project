"""JSONL run logging and simple token-cost estimates."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT

DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "runs.jsonl"


def estimate_cost(
    model: str,
    tokens_in: int,
    tokens_out: int,
    pricing_dict: dict[str, dict[str, float]],
) -> float:
    """Estimate USD cost from per-1M-token pricing."""
    rates = pricing_dict.get(model) or {}
    input_rate = float(rates.get("input", 0.0))
    output_rate = float(rates.get("output", 0.0))
    return (int(tokens_in) * input_rate + int(tokens_out) * output_rate) / 1_000_000.0


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


class EventLogger:
    """Append one JSON object per line to ``logs/runs.jsonl``."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = (path or DEFAULT_LOG_PATH).resolve()

    def log(
        self,
        *,
        request_id: str,
        model: str,
        prompt_version: str,
        retrieval_method: str,
        n_matches: int,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        estimated_cost_usd: float,
        error: str | None,
        mode: str,
        ts: float | None = None,
        **extra: Any,
    ) -> None:
        record = {
            "request_id": request_id,
            "ts": time.time() if ts is None else ts,
            "model": model,
            "prompt_version": prompt_version,
            "retrieval_method": retrieval_method,
            "n_matches": int(n_matches),
            "latency_ms": float(latency_ms),
            "tokens_in": int(tokens_in),
            "tokens_out": int(tokens_out),
            "estimated_cost_usd": float(estimated_cost_usd),
            "error": error,
            "mode": mode,
        }
        if extra:
            record.update(extra)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
