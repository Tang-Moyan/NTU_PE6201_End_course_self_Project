"""Offline Responsible AI tests: adversarial cases + deterministic FakeClient checks."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from main.pipeline import analyze_message
from main.retrieve import load_quotes, retrieve
from main.schemas import parse_analysis_json

RAI_CASES_PATH = Path(__file__).with_name("rai_cases.json")
REQUIRED_CATEGORIES = {
    "prompt_injection",
    "false_attribution",
    "fabricated_quote",
    "ambiguity",
    "offensive_language",
    "privacy_pii",
    "long_input",
    "jailbreak_roleplay",
    "contradiction",
}


class FakeClient:
    """Deterministic offline client returning fixed JSON for pipeline tests."""

    def __init__(self, reply: str = "{}") -> None:
        self.reply = reply
        self.calls: list[dict] = []
        self._tokens = {"in": 11, "out": 7}
        self.model = "fake/rai-test-model"

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
        return self.generate(
            prompt,
            system=system,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    @property
    def tokens(self) -> dict[str, int]:
        return dict(self._tokens)


def load_rai_cases() -> list[dict]:
    payload = json.loads(RAI_CASES_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if not isinstance(cases, list):
        raise AssertionError("rai_cases.json must contain a list under 'cases'")
    return cases


def _valid_analysis_json(
    *,
    user_text: str,
    used_knowledge_base: bool,
    safety_flags: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "user_text": user_text,
            "intent": "rai_stub",
            "used_knowledge_base": used_knowledge_base,
            "matched_quotes": [],
            "interpretation": "Deterministic stub interpretation for RAI tests.",
            "confidence": 0.35,
            "uncertainty": "stub",
            "safety_flags": list(safety_flags or []),
            "evidence": ["offline rai stub"],
        }
    )


class TestRaiCatalog(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_rai_cases()

    def test_catalog_has_required_categories_and_size(self) -> None:
        self.assertGreaterEqual(len(self.cases), 16)
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        present = {case["category"] for case in self.cases}
        self.assertTrue(REQUIRED_CATEGORIES.issubset(present), present)
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertIn("id", case)
                self.assertIn("category", case)
                self.assertIn("text", case)
                self.assertIn("expected_behavior", case)
                self.assertTrue(case["text"].strip())
                self.assertTrue(case["expected_behavior"].strip())


class TestRaiRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quotes = load_quotes()
        cls.cases = load_rai_cases()
        cls.by_category = {
            category: [c for c in cls.cases if c["category"] == category]
            for category in REQUIRED_CATEGORIES
        }

    def test_fabricated_quotes_do_not_force_kb_hits(self) -> None:
        for case in self.by_category["fabricated_quote"]:
            with self.subTest(case=case["id"]):
                matches = retrieve(case["text"], self.quotes)
                self.assertEqual(
                    matches,
                    [],
                    f"fabricated text unexpectedly retrieved: {[m.entry.quote for m in matches]}",
                )

    def test_false_attribution_does_not_return_wrong_source_label(self) -> None:
        for case in self.by_category["false_attribution"]:
            with self.subTest(case=case["id"]):
                matches = retrieve(case["text"], self.quotes)
                # May hit the real quote, but the KB source must not invent a wrong work label.
                for match in matches:
                    self.assertIn(
                        match.entry.source,
                        {"Harry Potter", "Rick And Morty", "The Lord Of The Rings"},
                    )
                # If the Harry Potter choices quote is claimed as Rick and Morty, source stays HP.
                if "It is not our abilities that show what we truly are" in case["text"]:
                    self.assertTrue(matches)
                    self.assertEqual(matches[0].entry.source, "Harry Potter")
                if "All we have to decide is what to do with the time that is given to us" in case["text"]:
                    self.assertTrue(matches)
                    self.assertEqual(matches[0].entry.source, "The Lord Of The Rings")

    def test_long_input_without_quote_stays_a_miss(self) -> None:
        for case in self.by_category["long_input"]:
            with self.subTest(case=case["id"]):
                matches = retrieve(case["text"], self.quotes)
                self.assertEqual(matches, [])

    def test_contradiction_with_embedded_quote_can_still_retrieve(self) -> None:
        case = next(c for c in self.by_category["contradiction"] if "wizard is never late" in c["text"])
        matches = retrieve(case["text"], self.quotes)
        found = {m.entry.quote for m in matches}
        self.assertIn(
            "A wizard is never late, Frodo Baggins. Nor is he early; he arrives precisely when he means to.",
            found,
        )


class TestRaiPipelineStub(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_rai_cases()

    def test_fake_client_json_parses_for_all_rai_cases(self) -> None:
        for case in self.cases:
            with self.subTest(case=case["id"]):
                reply = _valid_analysis_json(
                    user_text=case["text"],
                    used_knowledge_base=False,
                    safety_flags=[],
                )
                client = FakeClient(reply=reply)
                result = analyze_message(case["text"], client)
                self.assertIsNotNone(result.result)
                parsed = parse_analysis_json(result.raw_reply)
                self.assertEqual(parsed.intent, "rai_stub")
                self.assertFalse(parsed.used_knowledge_base)

    def test_injection_stub_can_surface_safety_flags(self) -> None:
        injection_cases = [c for c in self.cases if c["category"] == "prompt_injection"]
        self.assertTrue(injection_cases)
        for case in injection_cases:
            with self.subTest(case=case["id"]):
                reply = _valid_analysis_json(
                    user_text=case["text"],
                    used_knowledge_base=False,
                    safety_flags=["injection"],
                )
                client = FakeClient(reply=reply)
                result = analyze_message(case["text"], client)
                self.assertIsNotNone(result.result)
                self.assertIn("injection", result.result.safety_flags)

    def test_privacy_and_abuse_stubs_flag_safely(self) -> None:
        mapping = {
            "privacy_pii": "privacy",
            "offensive_language": "abuse",
            "jailbreak_roleplay": "injection",
        }
        for category, flag in mapping.items():
            case = next(c for c in self.cases if c["category"] == category)
            with self.subTest(category=category):
                reply = _valid_analysis_json(
                    user_text=case["text"],
                    used_knowledge_base=False,
                    safety_flags=[flag],
                )
                client = FakeClient(reply=reply)
                result = analyze_message(case["text"], client)
                self.assertIsNotNone(result.result)
                self.assertIn(flag, result.result.safety_flags)


if __name__ == "__main__":
    unittest.main()
