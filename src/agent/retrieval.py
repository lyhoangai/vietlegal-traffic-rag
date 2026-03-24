"""Retrieval, reranking, and official-source web fallback nodes."""

from __future__ import annotations

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.agent.shared import (
    _all_docs_cache,
    _dedupe_documents,
    _dedupe_sources,
    _env_flag,
    _get_db,
    _is_out_of_scope_query,
    _normalize_text,
    _query_requests_web_confirmation,
    _query_requests_web_freshness,
)
from src.agent.state import AgentState
from src.web_search import format_web_docs, format_web_sources, search_official_web

def _get_all_documents(db: Chroma, collection: str) -> list:
    if collection not in _all_docs_cache:
        raw = db.get(include=["documents", "metadatas"])
        documents = raw.get("documents", []) or []
        metadatas = raw.get("metadatas", []) or []
        _all_docs_cache[collection] = [
            Document(page_content=text or "", metadata=meta or {})
            for text, meta in zip(documents, metadatas)
            if text
        ]
    return _all_docs_cache[collection]


def _build_keyword_hints(state: AgentState) -> tuple[list, list]:
    query = _normalize_text(state.get("user_query", ""))
    entities = state.get("entities", {}) or {}
    violation = _normalize_text(entities.get("violation_type", ""))
    phrases = []

    if "den do" in violation or "den tin hieu" in violation:
        phrases.extend(
            [
                "khong chap hanh hieu lenh cua den tin hieu giao thong",
                "hieu lenh cua den tin hieu giao thong",
                "den tin hieu giao thong",
            ]
        )
    if "toc do" in violation:
        phrases.extend(["toc do toi da", "toc do khai thac", "khoang cach an toan"])

    stop = {
        "bao",
        "nhieu",
        "toi",
        "da",
        "la",
        "cho",
        "mot",
        "nguoi",
        "xe",
        "o",
        "to",
        "may",
        "trong",
        "khu",
        "dan",
        "cu",
    }
    tokens = [token for token in query.split() if len(token) >= 4 and token not in stop]
    return phrases, tokens


def _keyword_rescue_docs(db: Chroma, collection: str, state: AgentState, k: int = 8) -> list:
    phrases, tokens = _build_keyword_hints(state)
    if not phrases and not tokens:
        return []

    scored = []
    for doc in _get_all_documents(db, collection):
        norm = _normalize_text(doc.page_content)
        score = 0
        for phrase in phrases:
            if phrase in norm:
                score += 8
        for token in tokens:
            if token in norm:
                score += 1
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:k]]

def _should_use_web_fallback(state: AgentState) -> bool:
    if _is_out_of_scope_query(state.get("user_query", "")):
        return False
    if state.get("intent") not in {"penalty", "law", "speed"}:
        return False

    if _query_requests_web_confirmation(state.get("user_query", "")):
        return True

    if state.get("intent") == "penalty":
        return True

    local_docs = state.get("reranked_docs") or state.get("retrieved_docs") or []
    if _query_requests_web_freshness(state.get("user_query", "")):
        return True
    if not local_docs:
        return True
    if state.get("intent") in {"law", "speed"} and len(local_docs) < 2:
        return True
    return False


COLLECTION_ROUTING = {
    "penalty": "traffic_penalties",
    "law": "traffic_law",
    "speed": "traffic_speed",
    "general": "traffic_law",
}


async def query_router(state: AgentState) -> AgentState:
    """Choose the Chroma collection based on detected intent."""
    intent = state.get("intent", "general")
    collection = COLLECTION_ROUTING.get(intent, "traffic_law")
    return {**state, "collection_used": collection}


async def retriever(state: AgentState) -> AgentState:
    """Retrieve top documents from the selected collection."""
    collection = state.get("collection_used", "traffic_law")
    db = _get_db(collection)

    entities = state.get("entities", {})
    enriched_query = state["user_query"]
    if entities.get("vehicle_type"):
        enriched_query = f"{entities['vehicle_type']} {enriched_query}"
    if entities.get("violation_type"):
        enriched_query = f"{entities['violation_type']} {enriched_query}"

    try:
        vector_docs = db.similarity_search(enriched_query, k=20)
        keyword_docs = _keyword_rescue_docs(db, collection, state, k=8)
        docs = _dedupe_documents(keyword_docs + vector_docs)[:20]
    except Exception as exc:
        print(f"[RETRIEVER WARNING] Retrieval failed: {exc}")
        return {**state, "retrieved_docs": [], "sources": []}

    return {
        **state,
        "retrieved_docs": [d.page_content for d in docs],
        "sources": [
            f"{d.metadata.get('source_file', '?')} trang {d.metadata.get('page', 0) + 1}"
            for d in docs
        ],
    }


def _get_reranker():
    """Lazy-load the cross-encoder model."""
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None


_reranker_model = None


async def reranker(state: AgentState) -> AgentState:
    """Rerank retrieved docs with a Cross-Encoder and keep top 5."""
    global _reranker_model
    docs = state.get("retrieved_docs", [])

    if not _env_flag("ENABLE_RERANKER", default=True):
        return {**state, "reranked_docs": docs[:5]}

    if len(docs) <= 5:
        return {**state, "reranked_docs": docs}

    if _reranker_model is None:
        _reranker_model = _get_reranker()

    if _reranker_model is None:
        return {**state, "reranked_docs": docs[:5]}

    query = state["user_query"]
    pairs = [(query, doc) for doc in docs]
    scores = _reranker_model.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top5 = [doc for _, doc in ranked[:5]]
    return {**state, "reranked_docs": top5}


async def web_searcher(state: AgentState) -> AgentState:
    """Fetch official-source web snippets when local evidence is weak or freshness is requested."""
    if not _env_flag("ENABLE_WEB_FALLBACK", default=True):
        return {**state, "web_docs": []}

    if not _should_use_web_fallback(state):
        return {**state, "web_docs": []}

    web_query = _normalize_text(state.get("user_query", "")) or state.get("user_query", "")
    web_entities = {}
    for key, value in (state.get("entities", {}) or {}).items():
        if isinstance(value, str):
            web_entities[key] = _normalize_text(value) or value
        else:
            web_entities[key] = value

    results = await search_official_web(
        web_query,
        intent=state.get("intent", "general"),
        entities=web_entities,
        max_results=3,
    )
    if not results:
        return {**state, "web_docs": []}

    sources = _dedupe_sources(format_web_sources(results) + (state.get("sources") or []))
    return {
        **state,
        "web_docs": format_web_docs(results),
        "sources": sources,
    }
