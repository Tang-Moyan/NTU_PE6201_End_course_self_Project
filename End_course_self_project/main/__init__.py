"""Quote-aware chat interpretation application package.

Modules
-------
config
    Load ``config.yaml`` and environment overrides.
retrieve
    Load ``Knowledge_base`` quotes; fuzzy match helper types.
baselines
    TF-IDF / hashing-embedding retrievers, source classifier, ``get_retriever``.
pipeline
    End-to-end analyze path: retrieve → prompt → LLM → structured result + logs.
agent
    Bounded tool-using loop (search / details / finish) with pipeline fallback.
schemas
    Pydantic ``AnalysisResult`` and JSON parse helpers.
observability
    JSONL ``EventLogger`` and rough USD cost estimates.
API_model
    OpenRouter OpenAI-compatible client.
"""
