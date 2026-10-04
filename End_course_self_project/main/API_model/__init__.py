"""Model API clients.

Each provider lives in its own module. OpenRouter is `openrouter.py`.
"""

from .openrouter import DEFAULT_MODEL, OpenRouterClient, resolve_api_key

__all__ = ["DEFAULT_MODEL", "OpenRouterClient", "resolve_api_key"]
