"""Structured analysis schemas and JSON parsing helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


class QuoteEvidence(BaseModel):
    source: str
    quote: str
    matched_text: str
    score: float
    meaning: str
    chat_usage: str


class AnalysisResult(BaseModel):
    user_text: str
    intent: str
    used_knowledge_base: bool
    matched_quotes: list[QuoteEvidence] = Field(default_factory=list)
    interpretation: str
    confidence: float
    uncertainty: str
    safety_flags: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _confidence_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return float(value)


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json_blob(text: str) -> str:
    cleaned = text.strip()
    fence = _FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def parse_analysis_json(text: str) -> AnalysisResult:
    """Parse model output into ``AnalysisResult``, tolerating markdown fences."""
    blob = _extract_json_blob(text)
    try:
        payload: Any = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model output is not valid JSON: {exc}") from exc
    return AnalysisResult.model_validate(payload)
