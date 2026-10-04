"""Tests for TF-IDF / hashing retrieval baselines and the source classifier."""

from __future__ import annotations

import unittest

from main.baselines import (
    DiscriminativeSourceClassifier,
    EmbeddingRetriever,
    LocalHashingRetriever,
    TfidfRetriever,
    get_retriever,
)
from main.retrieve import load_quotes


class TestBaselineRetrievers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quotes = load_quotes()

    def test_tfidf_hits_you_shall_not_pass(self) -> None:
        retriever = TfidfRetriever(self.quotes, threshold=0.25)
        matches = retriever.search("You shall not pass.", limit=5)
        self.assertTrue(matches)
        self.assertTrue(
            any("You shall not pass." in (m.entry.quote, m.matched_text) for m in matches)
        )

    def test_embedding_hits_you_shall_not_pass(self) -> None:
        retriever = EmbeddingRetriever(self.quotes, threshold=0.25)
        matches = retriever.search("You shall not pass.", limit=5)
        self.assertTrue(matches)
        self.assertTrue(
            any("You shall not pass." in (m.entry.quote, m.matched_text) for m in matches)
        )
        self.assertIs(EmbeddingRetriever, LocalHashingRetriever)

    def test_near_miss_empty_at_high_threshold(self) -> None:
        near = "You shall not do that."
        for factory in (TfidfRetriever, EmbeddingRetriever):
            with self.subTest(retriever=factory.__name__):
                retriever = factory(self.quotes, threshold=0.85)
                matches = retriever.search(near, limit=5)
                self.assertEqual(matches, [])

    def test_get_retriever_factory(self) -> None:
        fuzzy = get_retriever("fuzzy", quotes=self.quotes)
        tfidf = get_retriever("tfidf", quotes=self.quotes)
        embedding = get_retriever("embedding", quotes=self.quotes)
        self.assertTrue(hasattr(fuzzy, "search"))
        self.assertTrue(hasattr(tfidf, "search"))
        self.assertTrue(hasattr(embedding, "search"))
        self.assertTrue(fuzzy.search("You shall not pass."))


class TestSourceClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quotes = load_quotes()
        cls.clf = DiscriminativeSourceClassifier().fit_from_kb(quotes=cls.quotes)

    def test_predicts_harry_potter_quote(self) -> None:
        quote = (
            "It is not our abilities that show what we truly are. It is our choices."
        )
        label, probs = self.clf.predict_source(quote)
        self.assertEqual(label, "Harry Potter")
        self.assertIn("Harry Potter", probs)
        self.assertGreater(probs["Harry Potter"], 0.5)

    def test_predict_has_quote_heuristic(self) -> None:
        self.assertTrue(self.clf.predict_has_quote("You shall not pass."))
        self.assertFalse(
            self.clf.predict_has_quote(
                "completely unrelated astronomy lunch schedule",
                threshold=0.9,
            )
        )


if __name__ == "__main__":
    unittest.main()
