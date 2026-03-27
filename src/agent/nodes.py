"""LangGraph nodes for VietLegal's retrieval and answer pipeline."""

from __future__ import annotations

import json
import os

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.agent.rule_based import (
    _build_conservative_answer,
    _build_rule_based_penalty_answer,
    _build_rule_based_speed_answer,
    _build_scope_limited_answer,
    _needs_conservative_fallback,
)
from src.agent.state import AgentState
from src.agent.text_utils import (
    _history_context,
    _history_text,
    _infer_entities_from_query,
    _infer_intent_from_context,
    _is_out_of_scope_query,
    _merge_entities_with_history,
    _normalize_text,
    _query_requests_web_confirmation,
    _query_requests_web_freshness,
)
from src.embeddings import get_embedding_function
from src.llm import invoke_with_fallback
from src.web_search import format_web_docs, format_web_sources, search_official_web

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

_dbs: dict = {}
_all_docs_cache: dict = {}


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _answer_with_sources(answer: str, sources: list, limit: int = 3) -> str:
    return answer


def _get_db(collection: str) -> Chroma:
    if collection not in _dbs:
        col_path = os.path.join(CHROMA_PATH, collection)
        embeddings = get_embedding_function()
        _dbs[collection] = Chroma(
            persist_directory=col_path,
            embedding_function=embeddings,
            collection_name=collection,
        )
    return _dbs[collection]


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


def _dedupe_documents(docs: list) -> list:
    unique = []
    seen = set()
    for doc in docs:
        key = (
            doc.metadata.get("source_file", ""),
            doc.metadata.get("page", -1),
            doc.page_content[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _dedupe_sources(sources: list) -> list:
    unique = []
    seen = set()
    for source in sources:
        if source in seen:
            continue
        seen.add(source)
        unique.append(source)
    return unique


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


INTENT_PROMPT = """Phân tích câu hỏi pháp luật giao thông sau và trích xuất thông tin.

Lịch sử hội thoại gần đây:
{history_context}

Câu hỏi hiện tại: {query}

Trả về JSON hợp lệ với các trường sau (dùng null nếu không có):
{{
  "intent": "penalty" | "law" | "speed" | "general",
  "vehicle_type": "ô tô" | "xe máy" | "xe tải" | null,
  "violation_type": string | null,
  "alcohol_level": string | null,
  "speed_value": string | null,
  "needs_clarification": true | false,
  "missing_field": "vehicle_type" | null
}}

Chỉ trả về JSON, không giải thích thêm."""


async def intent_analyzer(state: AgentState) -> AgentState:
    """Use the LLM to extract structured entities from the user query."""
    if _is_out_of_scope_query(state["user_query"]):
        return {
            **state,
            "intent": "general",
            "entities": _merge_entities_with_history(
                state["user_query"],
                _history_text(state.get("messages", [])),
                {},
            ),
            "needs_clarification": False,
            "clarification_question": "",
        }

    history_context = _history_context(state.get("messages", []))
    prompt = INTENT_PROMPT.format(query=state["user_query"], history_context=history_context)
    try:
        raw = await invoke_with_fallback(prompt, state)
    except Exception:
        raw = ""

    try:
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        extracted = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        extracted = {"intent": "general", "needs_clarification": False}

    entities = {
        "vehicle_type": extracted.get("vehicle_type"),
        "violation_type": extracted.get("violation_type"),
        "alcohol_level": extracted.get("alcohol_level"),
        "speed_value": extracted.get("speed_value"),
    }
    history_text = _history_text(state.get("messages", []))
    entities = _merge_entities_with_history(state["user_query"], history_text, entities)

    intent = extracted.get("intent", "general")
    if intent == "general":
        intent = _infer_intent_from_context(state["user_query"], history_text, entities)
    needs_clarification = extracted.get("needs_clarification", False)
    if entities.get("vehicle_type"):
        needs_clarification = False
    if intent == "speed":
        needs_clarification = False

    return {
        **state,
        "intent": intent,
        "entities": entities,
        "needs_clarification": needs_clarification,
        "clarification_question": (
            "Để tra cứu chính xác, anh/chị đang đi xe gì? (ô tô / xe máy / xe tải)"
            if needs_clarification
            else ""
        ),
    }


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
        "retrieved_docs": [doc.page_content for doc in docs],
        "sources": [
            f"{doc.metadata.get('source_file', '?')} trang {doc.metadata.get('page', 0) + 1}"
            for doc in docs
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

    ranked = sorted(zip(scores, docs), key=lambda item: item[0], reverse=True)
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


async def clarifier(state: AgentState) -> AgentState:
    """Set the answer to the clarification question."""
    return {
        **state,
        "answer": state.get("clarification_question", "Anh/chị đi xe gì ạ?"),
    }


def get_generation_docs(state: AgentState) -> list:
    local_docs = state.get("reranked_docs") or state.get("retrieved_docs", [])
    web_docs = state.get("web_docs", [])
    return list(local_docs) + list(web_docs)


def _collect_evidence_docs(state: AgentState) -> list[str]:
    evidence_docs: list[str] = []
    seen_text = set()
    for doc in list(state.get("reranked_docs") or []) + list(state.get("retrieved_docs") or []) + list(
        state.get("web_docs") or []
    ):
        text = doc.page_content if isinstance(doc, Document) else doc
        if not text or text in seen_text:
            continue
        seen_text.add(text)
        evidence_docs.append(text)
    return evidence_docs


def build_early_answer(state: AgentState, docs: list | None = None) -> tuple[str | None, float | None]:
    docs = docs if docs is not None else get_generation_docs(state)
    if _is_out_of_scope_query(state.get("user_query", "")):
        return _build_scope_limited_answer(), 0.2

    local_docs = state.get("reranked_docs") or state.get("retrieved_docs") or []
    evidence_docs = _collect_evidence_docs(state)

    if local_docs and state.get("intent") == "penalty":
        rule_answer, rule_confidence = _build_rule_based_penalty_answer(state, evidence_docs)
        if rule_answer is not None:
            return rule_answer, rule_confidence

    if state.get("intent") == "speed":
        speed_answer, speed_confidence = _build_rule_based_speed_answer(state, evidence_docs)
        if speed_answer is not None:
            return speed_answer, speed_confidence

    if docs:
        return None, None
    if state.get("intent") == "speed":
        return (
            "Không đủ căn cứ trong tài liệu hiện có để kết luận về tốc độ hoặc khoảng cách an toàn.",
            0.3,
        )
    return "Không đủ căn cứ trong tài liệu hiện có.", 0.3


def _build_entity_context(state: AgentState) -> str:
    entities = state.get("entities", {})
    entity_ctx_parts = []
    if entities.get("vehicle_type"):
        entity_ctx_parts.append(f"Phương tiện: {entities['vehicle_type']}")
    if entities.get("violation_type"):
        entity_ctx_parts.append(f"Vi phạm: {entities['violation_type']}")
    if entities.get("alcohol_level"):
        entity_ctx_parts.append(f"Nồng độ cồn: {entities['alcohol_level']}")
    return "; ".join(entity_ctx_parts) if entity_ctx_parts else "Không xác định"


GENERATOR_PROMPT = """Bạn là Trợ lý Pháp luật Giao thông Việt Nam. Dựa vào tài liệu pháp luật sau, hãy trả lời câu hỏi một cách chính xác, ngắn gọn, và trích dẫn Điều/Khoản cụ thể.

Lịch sử hội thoại gần đây:
{history_context}

Thông tin bổ sung: {entity_context}
Câu hỏi: {query}

Tài liệu cục bộ:
{local_context}

Nguồn web chính thống (fallback khi cần cập nhật):
{web_context}

Yêu cầu:
- Trả lời bằng tiếng Việt
- Nêu rõ mức phạt tiền (nếu có)
- Trích dẫn số Điều, Khoản, hoặc văn bản cụ thể
- Ưu tiên tài liệu cục bộ nếu đã có điều khoản rõ ràng
- Chỉ dùng nguồn web chính thống để bổ sung cập nhật hoặc khi local RAG không đủ
- Không lặp lại danh sách nguồn tham khảo trong thân bài; chỉ nêu căn cứ ngắn gọn khi cần
- Tuyệt đối không đoán hoặc suy luận ngoài tài liệu
- Nếu không tìm được điều khoản rõ ràng: trả lời đúng câu 'Không đủ căn cứ trong tài liệu hiện có.'"""


def build_generator_prompt(state: AgentState, docs: list | None = None) -> str:
    docs = docs if docs is not None else get_generation_docs(state)
    local_docs = list(state.get("reranked_docs") or state.get("retrieved_docs", []))
    web_docs = list(state.get("web_docs", []))
    local_context = "\n\n".join(local_docs[:5]) or "Không có"
    web_context = "\n\n".join(web_docs[:3]) or "Không có"
    return GENERATOR_PROMPT.format(
        history_context=_history_context(state.get("messages", [])),
        entity_context=_build_entity_context(state),
        query=state["user_query"],
        local_context=local_context,
        web_context=web_context,
    )


def finalize_generated_answer(
    state: AgentState,
    answer: str,
    docs: list | None = None,
) -> tuple[str, float]:
    docs = docs if docs is not None else get_generation_docs(state)
    local_docs = state.get("reranked_docs") or state.get("retrieved_docs", [])
    web_docs = state.get("web_docs", [])
    if len(local_docs) >= 3:
        confidence = 0.85
    elif local_docs and web_docs:
        confidence = 0.75
    elif local_docs:
        confidence = 0.65
    elif web_docs:
        confidence = 0.6
    else:
        confidence = 0.5

    if state.get("intent") in {"penalty", "law", "speed"} and _needs_conservative_fallback(answer):
        answer = _build_conservative_answer(state.get("intent", "general"))
        confidence = min(confidence, 0.4)
    elif confidence < 0.7:
        answer += "\n\nLưu ý: vui lòng kiểm tra lại với văn bản pháp luật chính thức."
    return answer, confidence


async def generator(state: AgentState) -> AgentState:
    """Generate the final answer using reranked docs."""
    docs = get_generation_docs(state)
    early_answer, early_confidence = build_early_answer(state, docs)
    if early_answer is not None:
        return {**state, "answer": early_answer, "confidence": early_confidence}

    prompt = build_generator_prompt(state, docs)
    answer = await invoke_with_fallback(prompt, state)
    answer, confidence = finalize_generated_answer(state, answer, docs)
    return {**state, "answer": answer, "confidence": confidence}
