"""Load project ``config.yaml`` from the repository root."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class RetrievalConfig:
    method: str = "fuzzy"
    limit: int = 5


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 3
    timeout_seconds: int = 60


@dataclass(frozen=True)
class AppConfig:
    model: str = "openai/gpt-4o-mini"
    alternate_models: tuple[str, ...] = ("openai/gpt-4o-mini", "qwen/qwen-2.5-7b-instruct")
    temperature: float = 0.0
    max_new_tokens: int = 450
    knowledge_dir: str = "Knowledge_base"
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    pricing_per_1m_tokens: dict[str, dict[str, float]] = field(default_factory=dict)
    prompt_version: str = "v2-structured"

    def knowledge_path(self, root: Path | None = None) -> Path:
        base = root or PROJECT_ROOT
        path = Path(self.knowledge_dir)
        return path if path.is_absolute() else (base / path).resolve()


def _as_dict(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def load_config(path: Path | None = None) -> AppConfig:
    """Load ``config.yaml`` and return a typed ``AppConfig``."""
    config_path = (path or DEFAULT_CONFIG_PATH).resolve()
    raw = _as_dict(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})

    retrieval_raw = raw.get("retrieval") or {}
    agent_raw = raw.get("agent") or {}
    pricing = raw.get("pricing_per_1m_tokens") or {}

    return AppConfig(
        model=str(raw.get("model", AppConfig.model)),
        alternate_models=tuple(raw.get("alternate_models") or AppConfig().alternate_models),
        temperature=float(raw.get("temperature", 0.0)),
        max_new_tokens=int(raw.get("max_new_tokens", 450)),
        knowledge_dir=str(raw.get("knowledge_dir", "Knowledge_base")),
        retrieval=RetrievalConfig(
            method=str(retrieval_raw.get("method", "fuzzy")),
            limit=int(retrieval_raw.get("limit", 5)),
        ),
        agent=AgentConfig(
            max_steps=int(agent_raw.get("max_steps", 3)),
            timeout_seconds=int(agent_raw.get("timeout_seconds", 60)),
        ),
        pricing_per_1m_tokens={
            str(model): {
                "input": float((rates or {}).get("input", 0.0)),
                "output": float((rates or {}).get("output", 0.0)),
            }
            for model, rates in pricing.items()
        },
        prompt_version=str(raw.get("prompt_version", "v2-structured")),
    )


def get_pricing(config: AppConfig | None = None) -> dict[str, dict[str, float]]:
    """Return the pricing map from config (or a freshly loaded config)."""
    cfg = config or load_config()
    return dict(cfg.pricing_per_1m_tokens)
