"""OpenRouter chat completions.

The client follows the course starter: an OpenAI-compatible client pointed at
OpenRouter, temperature 0 by default, and a running token count.

The key is never stored in code. Resolution order:

1. The ``api_key`` argument.
2. ``MY_PRIVATE_OPENROUTER_KEY`` or ``OPENROUTER_API_KEY``.
3. A Colab secret named ``OPENROUTER_API_KEY``, when that runtime exists.
4. A hidden prompt, only if this process has an interactive terminal.
"""

from __future__ import annotations

import getpass
import os
import sys

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def resolve_api_key(api_key: str | None = None) -> str:
    """Return an OpenRouter key, or raise if none can be obtained."""
    if api_key:
        return api_key

    key = os.environ.get("MY_PRIVATE_OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key

    try:
        from google.colab import userdata

        key = userdata.get("OPENROUTER_API_KEY")
        if key:
            return key
    except Exception:
        pass

    if sys.stdin.isatty():
        return getpass.getpass("Paste your OpenRouter API key (hidden): ")

    raise RuntimeError(
        "OpenRouter API key not found. Set OPENROUTER_API_KEY or MY_PRIVATE_OPENROUTER_KEY."
    )


class OpenRouterClient:
    """One OpenRouter model, with its own token counter.

    ``temperature`` defaults to 0 so repeated calls stay comparable.
    Pass another model id, such as ``qwen/qwen-2.5-7b-instruct``, to switch models.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 900,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._tokens = {"in": 0, "out": 0}
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=resolve_api_key(api_key),
        )

    @property
    def tokens(self) -> dict[str, int]:
        """Prompt and completion tokens used by this client so far."""
        return dict(self._tokens)

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        *,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Send one chat completion and return the stripped text."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_new_tokens if max_new_tokens is None else max_new_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        if usage:
            self._tokens["in"] += usage.prompt_tokens
            self._tokens["out"] += usage.completion_tokens
        content = response.choices[0].message.content
        return (content or "").strip()

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Ask for a JSON object and return the raw model text for the caller to parse."""
        json_instruction = (
            "Respond with a single valid JSON object only. "
            "Do not wrap it in commentary. Markdown fences are acceptable but unnecessary. "
            "Never truncate fields mid-string; finish the full JSON object."
        )
        if system:
            combined = f"{system.rstrip()}\n\n{json_instruction}"
        else:
            combined = json_instruction
        return self.generate(
            prompt,
            system=combined,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

    def ping(self) -> str:
        """Cheap connectivity check. Expect the model to answer ``ready``."""
        return self.generate("Reply with exactly the word: ready")
