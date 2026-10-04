"""Baseline retrievers and a discriminative source classifier for course comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

from .retrieve import QuoteEntry, QuoteMatch, load_quotes, retrieve

DEFAULT_TFIDF_THRESHOLD = 0.25
DEFAULT_EMBEDDING_THRESHOLD = 0.25
DEFAULT_NO_MATCH_THRESHOLD = 0.20


class SupportsSearch(Protocol):
    def search(self, text: str, limit: int = 5) -> list[QuoteMatch]: ...


class FuzzyRetriever:
    """Thin wrapper around the existing fuzzy ``retrieve()`` function."""

    def __init__(self, quotes: tuple[QuoteEntry, ...] | None = None) -> None:
        self.quotes = quotes if quotes is not None else load_quotes()

    def search(self, text: str, limit: int = 5) -> list[QuoteMatch]:
        return retrieve(text, self.quotes, limit=limit)


class TfidfRetriever:
    """TF-IDF + cosine similarity over the quote and variation corpus."""

    def __init__(
        self,
        quotes: tuple[QuoteEntry, ...] | None = None,
        *,
        threshold: float = DEFAULT_TFIDF_THRESHOLD,
    ) -> None:
        self.quotes = quotes if quotes is not None else load_quotes()
        self.threshold = float(threshold)
        self._candidates: list[tuple[str, QuoteEntry]] = []
        for entry in self.quotes:
            seen: set[str] = set()
            for candidate in (entry.quote, *entry.variations):
                key = candidate.strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                self._candidates.append((candidate, entry))

        corpus = [text for text, _ in self._candidates] or [""]
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, text: str, limit: int = 5) -> list[QuoteMatch]:
        if not text or not text.strip():
            raise ValueError("User text is empty")
        if not self._candidates:
            return []

        query = self._vectorizer.transform([text])
        scores = cosine_similarity(query, self._matrix).ravel()
        return _top_matches(self._candidates, scores, threshold=self.threshold, limit=limit)


class LocalHashingRetriever:
    """Local hashing embedding proxy for course comparison (avoids paid embedding API).

    Uses sklearn ``HashingVectorizer`` bag-of-words hashing + cosine similarity as a
    lightweight semantic baseline. This is not a neural embedding service.
    """

    def __init__(
        self,
        quotes: tuple[QuoteEntry, ...] | None = None,
        *,
        threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
        n_features: int = 2**14,
    ) -> None:
        self.quotes = quotes if quotes is not None else load_quotes()
        self.threshold = float(threshold)
        self._candidates: list[tuple[str, QuoteEntry]] = []
        for entry in self.quotes:
            seen: set[str] = set()
            for candidate in (entry.quote, *entry.variations):
                key = candidate.strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                self._candidates.append((candidate, entry))

        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
        )
        corpus = [text for text, _ in self._candidates] or [""]
        self._matrix = self._vectorizer.transform(corpus)

    def search(self, text: str, limit: int = 5) -> list[QuoteMatch]:
        if not text or not text.strip():
            raise ValueError("User text is empty")
        if not self._candidates:
            return []

        query = self._vectorizer.transform([text])
        scores = cosine_similarity(query, self._matrix).ravel()
        return _top_matches(self._candidates, scores, threshold=self.threshold, limit=limit)


# Course-facing alias: embedding baseline without an external embedding API.
EmbeddingRetriever = LocalHashingRetriever


class DiscriminativeSourceClassifier:
    """Sklearn TF-IDF + LogisticRegression classifier over knowledge-base sources."""

    def __init__(self, *, no_match_threshold: float = DEFAULT_NO_MATCH_THRESHOLD) -> None:
        self.no_match_threshold = float(no_match_threshold)
        self.model: Pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        )
        self._fitted = False
        self._quote_vectorizer: TfidfVectorizer | None = None
        self._quote_matrix = None
        self._labels: tuple[str, ...] = ()

    def fit_from_kb(
        self,
        knowledge_dir: Path | None = None,
        quotes: tuple[QuoteEntry, ...] | None = None,
    ) -> DiscriminativeSourceClassifier:
        catalog = quotes if quotes is not None else load_quotes(knowledge_dir)
        texts: list[str] = []
        labels: list[str] = []
        for entry in catalog:
            for candidate in (entry.quote, *entry.variations):
                candidate = candidate.strip()
                if not candidate:
                    continue
                texts.append(candidate)
                labels.append(entry.source)

        if not texts:
            raise ValueError("No quote texts available to fit the source classifier")

        self.model.fit(texts, labels)
        self._fitted = True
        self._labels = tuple(sorted(set(labels)))

        self._quote_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self._quote_matrix = self._quote_vectorizer.fit_transform(texts)
        return self

    def predict_source(self, text: str) -> tuple[str, dict[str, float]]:
        self._ensure_fitted()
        if not text or not text.strip():
            raise ValueError("text is empty")

        proba = self.model.predict_proba([text])[0]
        classes = list(self.model.named_steps["clf"].classes_)
        probs = {str(label): float(score) for label, score in zip(classes, proba)}
        label = max(probs, key=probs.get)
        return label, probs

    def predict_has_quote(self, text: str, threshold: float | None = None) -> bool:
        """Heuristic: no match when max cosine similarity to any quote is below threshold."""
        self._ensure_fitted()
        if self._quote_vectorizer is None or self._quote_matrix is None:
            raise RuntimeError("quote similarity matrix is not available")
        if not text or not text.strip():
            return False

        cutoff = self.no_match_threshold if threshold is None else float(threshold)
        query = self._quote_vectorizer.transform([text])
        scores = cosine_similarity(query, self._quote_matrix).ravel()
        if scores.size == 0:
            return False
        return bool(float(np.max(scores)) >= cutoff)

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("classifier is not fitted; call fit_from_kb() first")


def get_retriever(
    method: str,
    quotes: tuple[QuoteEntry, ...] | None = None,
) -> SupportsSearch:
    """Factory returning an object with ``.search(text, limit=5)``."""
    catalog = quotes if quotes is not None else load_quotes()
    key = (method or "fuzzy").strip().lower()
    if key == "fuzzy":
        return FuzzyRetriever(catalog)
    if key == "tfidf":
        return TfidfRetriever(catalog)
    if key in {"embedding", "hashing", "local_hashing", "local-hashing"}:
        return LocalHashingRetriever(catalog)
    raise ValueError(f"Unknown retrieval method: {method!r}")


def _top_matches(
    candidates: list[tuple[str, QuoteEntry]],
    scores: np.ndarray,
    *,
    threshold: float,
    limit: int,
) -> list[QuoteMatch]:
    best_by_quote: dict[tuple[str, str], QuoteMatch] = {}
    for (candidate, entry), score in zip(candidates, scores):
        value = float(score)
        if value < threshold:
            continue
        key = (entry.source, entry.quote)
        current = best_by_quote.get(key)
        if current is None or value > current.score:
            best_by_quote[key] = QuoteMatch(entry=entry, score=value, matched_text=candidate)

    ranked = sorted(
        best_by_quote.values(),
        key=lambda match: (match.score, len(match.matched_text)),
        reverse=True,
    )
    return ranked[: max(0, int(limit))]
