"""Flow tests for quote retrieval and the analysis pipeline.

The model client is fake. These tests check which knowledge-base records are
selected, what is sent to the model, and that a hit is printed before the reply.
They do not call OpenRouter.
"""

from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from main.pipeline import _print_analysis, analyze_message
from main.retrieve import QuoteEntry, load_quotes, normalize, retrieve

CASES_PATH = Path(__file__).with_name("cases.json")
REQUIRED_TYPES = {
    "exact",
    "normalized",
    "variation",
    "paraphrase",
    "embedded",
    "multi",
    "cross_source",
    "no_hit",
    "near_miss",
}
CASES_PER_TYPE = 24
TOTAL_CASES = CASES_PER_TYPE * len(REQUIRED_TYPES)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "but",
    "for",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "that",
    "the",
    "to",
    "we",
    "you",
}


class FakeClient:
    """Records the prompt the pipeline would send, and returns a fixed reply."""

    def __init__(self, reply: str = "STUB ANALYSIS") -> None:
        self.reply = reply
        self.calls: list[dict] = []
        self._tokens = {"in": 11, "out": 7}
        self.model = "fake/test-model"

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
        return self.reply

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        json_instruction = (
            "Respond with a single valid JSON object only. "
            "Do not wrap it in commentary. Markdown fences are acceptable but unnecessary."
        )
        combined = f"{system.rstrip()}\n\n{json_instruction}" if system else json_instruction
        return self.generate(
            prompt,
            system=combined,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    @property
    def tokens(self) -> dict[str, int]:
        return dict(self._tokens)


def load_cases() -> list[dict]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if not isinstance(cases, list):
        raise AssertionError("cases.json must contain a list under 'cases'")
    return cases


def quote_index(entries: tuple[QuoteEntry, ...]) -> dict[str, QuoteEntry]:
    return {entry.quote: entry for entry in entries}


def content_words(text: str) -> set[str]:
    return {word for word in normalize(text).split() if word not in STOP_WORDS}


class TestCaseCatalog(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quotes = load_quotes()
        cls.by_quote = quote_index(cls.quotes)
        cls.cases = load_cases()

    def test_catalog_covers_result_types(self) -> None:
        self.assertEqual(len(self.cases), TOTAL_CASES)
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        present = {case["result_type"] for case in self.cases}
        self.assertEqual(present, REQUIRED_TYPES)
        for result_type in REQUIRED_TYPES:
            count = sum(case["result_type"] == result_type for case in self.cases)
            self.assertEqual(count, CASES_PER_TYPE, result_type)

    def test_case_labels_match_intended_structure(self) -> None:
        for case in self.cases:
            with self.subTest(case=case["id"]):
                result_type = case["result_type"]
                text = case["text"]
                expected = case["expected_quotes"]
                if result_type == "exact":
                    self.assertEqual(len(expected), 1)
                    self.assertEqual(text, expected[0])
                elif result_type == "normalized":
                    self.assertEqual(len(expected), 1)
                    quote = expected[0]
                    self.assertNotEqual(text, quote)
                    self.assertEqual(normalize(text), normalize(quote))
                elif result_type == "variation":
                    self.assertEqual(len(expected), 1)
                    entry = self.by_quote[expected[0]]
                    hay = normalize(text)
                    hits = [v for v in entry.variations if normalize(v) in hay]
                    self.assertTrue(hits, "text must contain a listed variation")
                    self.assertNotIn(entry.quote, text)
                elif result_type == "paraphrase":
                    self.assertEqual(len(expected), 1)
                    quote = expected[0]
                    entry = self.by_quote[quote]
                    self.assertNotEqual(text, quote)
                    for variation in entry.variations:
                        self.assertNotEqual(text.strip(), variation.strip())
                        self.assertNotEqual(normalize(text), normalize(variation))
                    self.assertNotIn(normalize(quote), normalize(text))
                elif result_type == "embedded":
                    self.assertEqual(len(expected), 1)
                    self.assertIn(expected[0], text)
                    self.assertNotEqual(text.strip(), expected[0].strip())
                elif result_type == "multi":
                    self.assertEqual(len(expected), 2)
                    sources = {self.by_quote[q].source for q in expected}
                    self.assertEqual(len(sources), 1)
                    for quote in expected:
                        self.assertIn(quote, text)
                elif result_type == "cross_source":
                    self.assertEqual(len(expected), 2)
                    sources = {self.by_quote[q].source for q in expected}
                    self.assertGreaterEqual(len(sources), 2)
                    for quote in expected:
                        self.assertIn(quote, text)
                elif result_type == "no_hit":
                    self.assertEqual(expected, [])
                elif result_type == "near_miss":
                    self.assertEqual(expected, [])
                    words = content_words(text)
                    overlap = max(
                        len(words & content_words(entry.quote)) for entry in self.quotes
                    )
                    self.assertGreaterEqual(
                        overlap,
                        1,
                        "near_miss must overlap a meaningful quote word",
                    )


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quotes = load_quotes()
        cls.cases = load_cases()

    def test_knowledge_base_loads_all_works(self) -> None:
        sources = {entry.source for entry in self.quotes}
        self.assertEqual(sources, {"Harry Potter", "Rick And Morty", "The Lord Of The Rings"})
        self.assertEqual(len(self.quotes), 72)

    def test_each_case_matches_the_expected_records(self) -> None:
        for case in self.cases:
            with self.subTest(case=case["id"]):
                matches = retrieve(case["text"], self.quotes)
                found = [match.entry.quote for match in matches]
                self.assertCountEqual(found, case["expected_quotes"])

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            retrieve("   ", self.quotes)

    def test_missing_knowledge_dir_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_quotes(Path("this-knowledge-dir-does-not-exist"))


class TestAnalysisFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases()

    def _case(self, result_type: str) -> dict:
        return next(case for case in self.cases if case["result_type"] == result_type)

    def test_hit_sends_record_notes_to_the_model(self) -> None:
        case = self._case("exact")
        client = FakeClient()
        result = analyze_message(case["text"], client)

        self.assertEqual(result.analysis, "STUB ANALYSIS")
        self.assertEqual([match.entry.quote for match in result.matches], case["expected_quotes"])
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        from main.config import load_config

        self.assertEqual(call["max_new_tokens"], load_config().max_new_tokens)
        self.assertIn("Do not invent a quotation", call["system"])
        prompt = call["prompt"]
        self.assertIn(case["text"], prompt)
        self.assertIn("Retrieved lines:", prompt)
        self.assertIn(case["expected_quotes"][0], prompt)
        self.assertIn("meaning:", prompt)
        self.assertIn("typical chat use:", prompt)
        self.assertIn(result.matches[0].entry.meaning, prompt)
        self.assertIn(result.matches[0].entry.chat_usage, prompt)

    def test_miss_asks_the_model_to_analyze_the_message_alone(self) -> None:
        case = self._case("no_hit")
        client = FakeClient()
        result = analyze_message(case["text"], client)

        self.assertEqual(result.matches, ())
        self.assertEqual(len(client.calls), 1)
        prompt = client.calls[0]["prompt"]
        self.assertIn("Retrieved lines: none.", prompt)
        self.assertIn("Interpret the message directly.", prompt)
        self.assertNotIn("meaning:", prompt)
        self.assertNotIn("typical chat use:", prompt)

    def test_multiple_hits_are_all_included(self) -> None:
        case = self._case("cross_source")
        client = FakeClient()
        result = analyze_message(case["text"], client)

        found = {match.entry.quote for match in result.matches}
        self.assertEqual(found, set(case["expected_quotes"]))
        sources = {match.entry.source for match in result.matches}
        self.assertGreaterEqual(len(sources), 2)
        prompt = client.calls[0]["prompt"]
        for quote in case["expected_quotes"]:
            self.assertIn(quote, prompt)

    def test_hit_is_printed_before_the_model_reply(self) -> None:
        case = self._case("exact")
        client = FakeClient(reply="MODEL SAYS THIS")
        buffer = StringIO()
        with redirect_stdout(buffer):
            _print_analysis(case["text"], client, knowledge_dir=None)

        output = buffer.getvalue()
        record_at = output.index("Matched 1 record")
        meaning_at = output.index("meaning:")
        reply_at = output.index("LLM reply:")
        self.assertLess(record_at, meaning_at)
        self.assertLess(meaning_at, reply_at)
        self.assertIn(case["expected_quotes"][0], output)
        self.assertIn("chat usage:", output)
        self.assertIn("variations:", output)
        self.assertIn("MODEL SAYS THIS", output[reply_at:])
        self.assertEqual(len(client.calls), 1)

    def test_miss_print_says_no_record_before_the_reply(self) -> None:
        case = self._case("near_miss")
        client = FakeClient(reply="PLAIN READING")
        buffer = StringIO()
        with redirect_stdout(buffer):
            _print_analysis(case["text"], client, knowledge_dir=None)

        output = buffer.getvalue()
        self.assertLess(output.index("Matched 0 records"), output.index("LLM reply:"))
        self.assertIn("PLAIN READING", output)
        self.assertNotIn("meaning:", output)


if __name__ == "__main__":
    unittest.main()
