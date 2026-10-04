"""Load knowledge-base quotes and find which ones a message is using."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "Knowledge_base"

# Fuzzy matches need a longer, distinctive span. Short lines must be contained.
_MIN_FUZZY_WORDS = 4
_FUZZY_THRESHOLD = 0.86
_MIN_CONTAINED_WORDS = 2

_TOKEN = re.compile(r"[a-z0-9']+")
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "`": "'"})


@dataclass(frozen=True)
class QuoteEntry:
    source: str
    quote: str
    meaning: str
    chat_usage: str
    variations: tuple[str, ...]


@dataclass(frozen=True)
class QuoteMatch:
    entry: QuoteEntry
    score: float
    matched_text: str


def normalize(text: str) -> str:
    """Lowercase and keep only words, so punctuation does not block a match."""
    cleaned = text.translate(_APOSTROPHES).lower()
    return " ".join(_TOKEN.findall(cleaned))


def load_quotes(directory: Path | None = None) -> tuple[QuoteEntry, ...]:
    """Read every ``quotes.json`` under the knowledge-base directory."""
    root = (directory or DEFAULT_KNOWLEDGE_DIR).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {root}")

    entries: list[QuoteEntry] = []
    files = sorted(root.glob("*/quotes.json"))
    if not files:
        raise FileNotFoundError(f"No quotes.json files under {root}")

    for path in files:
        source = path.parent.name.replace("_", " ")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a list of quotes")
        for index, item in enumerate(data):
            try:
                entries.append(
                    QuoteEntry(
                        source=source,
                        quote=str(item["quote"]),
                        meaning=str(item["meaning"]),
                        chat_usage=str(item["chat_usage"]),
                        variations=tuple(str(v) for v in item.get("variations") or ()),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path} item {index} is missing {exc}") from exc
    return tuple(entries)


def retrieve(text: str, quotes: tuple[QuoteEntry, ...] | None = None, *, limit: int = 5) -> list[QuoteMatch]:
    """Return quotes the message quotes or closely paraphrases, best match first."""
    if not text or not text.strip():
        raise ValueError("User text is empty")

    haystack = normalize(text)
    if not haystack:
        return []

    catalog = quotes if quotes is not None else load_quotes()
    found: list[QuoteMatch] = []
    for entry in catalog:
        best_score = 0.0
        best_text = ""
        for candidate in (entry.quote, *entry.variations):
            needle = normalize(candidate)
            score = _candidate_score(needle, haystack)
            if _prefer(score, needle, best_score, normalize(best_text)):
                best_score = score
                best_text = candidate
        if best_score > 0:
            found.append(QuoteMatch(entry=entry, score=best_score, matched_text=best_text))

    found.sort(key=lambda match: (match.score, len(normalize(match.matched_text))), reverse=True)
    return found[:limit]


def _prefer(score: float, needle: str, best_score: float, best_needle: str) -> bool:
    """Keep the more specific line when a short exact hit sits inside a longer near-hit."""
    if score <= 0:
        return False
    if not best_needle:
        return True
    words = len(needle.split())
    best_words = len(best_needle.split())
    if words >= best_words + 3 and score >= _FUZZY_THRESHOLD and score >= best_score - 0.05:
        return True
    if best_words >= words + 3 and best_score >= _FUZZY_THRESHOLD and best_score >= score - 0.05:
        return False
    if score > best_score:
        return True
    return score == best_score and words > best_words


def _candidate_score(needle: str, haystack: str) -> float:
    words = needle.split()
    if len(words) < _MIN_CONTAINED_WORDS:
        return 0.0
    if needle in haystack:
        return 1.0
    if len(words) < _MIN_FUZZY_WORDS:
        return 0.0
    score = _partial_ratio(needle, haystack)
    if score < _FUZZY_THRESHOLD:
        return 0.0
    return score


def _partial_ratio(needle: str, haystack: str) -> float:
    """Best similarity between ``needle`` and a similar-length slice of ``haystack``."""
    needle_words = needle.split()
    hay_words = haystack.split()
    if not needle_words or not hay_words:
        return 0.0
    if len(hay_words) <= len(needle_words) + 2:
        return SequenceMatcher(None, needle, haystack).ratio()

    best = 0.0
    shortest = max(1, len(needle_words) - 1)
    longest = len(needle_words) + 2
    for size in range(shortest, longest + 1):
        if size > len(hay_words):
            continue
        for start in range(0, len(hay_words) - size + 1):
            chunk = " ".join(hay_words[start : start + size])
            best = max(best, SequenceMatcher(None, needle, chunk).ratio())
    return best
