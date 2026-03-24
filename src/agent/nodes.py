"""Compatibility facade that re-exports the split agent modules."""

from __future__ import annotations

from src.agent.answers import (
    build_early_answer,
    build_generator_prompt,
    finalize_generated_answer,
    generator,
    get_generation_docs,
)
from src.agent.intent import (
    _history_context,
    _history_text,
    _infer_entities_from_query,
    _infer_intent_from_context,
    _merge_entities_with_history,
    _violation_topic,
    clarifier,
    intent_analyzer,
)
from src.agent.retrieval import (
    _get_reranker,
    _reranker_model,
    query_router,
    reranker,
    retriever,
    web_searcher,
)

__all__ = [
    "_get_reranker",
    "_history_context",
    "_history_text",
    "_infer_entities_from_query",
    "_infer_intent_from_context",
    "_merge_entities_with_history",
    "_reranker_model",
    "_violation_topic",
    "build_early_answer",
    "build_generator_prompt",
    "clarifier",
    "finalize_generated_answer",
    "generator",
    "get_generation_docs",
    "intent_analyzer",
    "query_router",
    "reranker",
    "retriever",
    "web_searcher",
]
