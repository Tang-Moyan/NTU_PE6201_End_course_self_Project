"""Agent loop tests with a fake OpenRouter-like client (no network)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from main.agent import run_agent
from main.config import load_config
from main.observability import EventLogger


ANALYSIS_JSON = {
    "user_text": "You shall not pass.",
    "intent": "refuse",
    "used_knowledge_base": True,
    "matched_quotes": [
        {
            "source": "The Lord Of The Rings",
            "quote": "You shall not pass.",
            "matched_text": "You shall not pass.",
            "score": 1.0,
            "meaning": "absolute boundary",
            "chat_usage": "firm refusal",
        }
    ],
    "interpretation": "The user is firmly refusing or blocking a request.",
    "confidence": 0.9,
    "uncertainty": "Tone could be joking.",
    "safety_flags": [],
    "evidence": ["Matched LOTR line You shall not pass."],
}


class FakeClient:
    """Stepwise replies: tool JSON, finish, then structured analysis JSON."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._tokens = {"in": 21, "out": 13}
        self.model = "fake/agent-model"
        self._generate_replies = [
            json.dumps({"tool": "search_quotes", "args": {"text": "You shall not pass."}}),
            json.dumps({"tool": "finish_analysis", "args": {}}),
            json.dumps(ANALYSIS_JSON),
        ]

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
                "kind": "generate",
            }
        )
        if not self._generate_replies:
            raise AssertionError("FakeClient has no remaining generate replies")
        return self._generate_replies.pop(0)

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
        reply = self.generate(
            prompt,
            system=combined,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        self.calls[-1]["kind"] = "generate_json"
        return reply

    @property
    def tokens(self) -> dict[str, int]:
        return dict(self._tokens)


class TestAgentLoop(unittest.TestCase):
    def test_agent_completes_with_fake_tool_steps(self) -> None:
        client = FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            logger = EventLogger(path=Path(tmp) / "runs.jsonl")
            result = run_agent(
                "You shall not pass.",
                client,
                config=load_config(),
                logger=logger,
                retrieval_method="fuzzy",
            )

        self.assertEqual(result.text, "You shall not pass.")
        self.assertTrue(result.matches)
        self.assertEqual(result.matches[0].entry.quote, "You shall not pass.")
        self.assertIsNotNone(result.result)
        assert result.result is not None
        self.assertEqual(result.result.intent, "refuse")
        self.assertTrue(result.result.used_knowledge_base)
        self.assertIn("firmly refusing", result.analysis.lower())

        # two tool turns + one final structured call
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(client.calls[0]["kind"], "generate")
        self.assertEqual(client.calls[1]["kind"], "generate")
        self.assertEqual(client.calls[2]["kind"], "generate_json")
        self.assertIn("search_quotes", client.calls[0]["system"])

    def test_invalid_tool_json_falls_back_to_pipeline(self) -> None:
        class BadThenOkClient(FakeClient):
            def __init__(self) -> None:
                super().__init__()
                self._generate_replies = [
                    "not-json-at-all",
                    json.dumps(ANALYSIS_JSON),
                ]

        client = BadThenOkClient()
        with tempfile.TemporaryDirectory() as tmp:
            logger = EventLogger(path=Path(tmp) / "runs.jsonl")
            result = run_agent(
                "You shall not pass.",
                client,
                config=load_config(),
                logger=logger,
            )

        self.assertTrue(result.matches)
        # bad tool generate + fallback pipeline generate_json->generate
        self.assertEqual(len(client.calls), 2)
        self.assertIsNotNone(result.result)


if __name__ == "__main__":
    unittest.main()
