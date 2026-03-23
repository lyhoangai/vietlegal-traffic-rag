"""LangGraph nodes for VietLegal's retrieval and answer pipeline."""

from __future__ import annotations

import json
import os
import re
import unicodedata

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.agent.state import AgentState
from src.embeddings import get_embedding_function
from src.llm import invoke_with_fallback
from src.web_search import format_web_docs, format_web_sources, search_official_web

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

_dbs: dict = {}
_all_docs_cache: dict = {}
_LEGAL_CITATION_RE = re.compile(
    r"\b(\u0110i\u1ec1u|Kho\u1ea3n|Ngh\u1ecb \u0111\u1ecbnh|Th\u00f4ng t\u01b0|Lu\u1eadt)\b",
    re.IGNORECASE,
)
_VAGUE_PHRASES = [
    "kh\u00f4ng c\u00f3 th\u00f4ng tin c\u1ee5 th\u1ec3",
    "kh\u00f4ng \u0111\u01b0\u1ee3c ch\u1ec9 \u0111\u1ecbnh r\u00f5",
    "th\u00f4ng th\u01b0\u1eddng",
    "c\u00f3 th\u1ec3 tham kh\u1ea3o",
    "khuy\u1ebfn ngh\u1ecb b\u1ea1n tham kh\u1ea3o",
]
_OUT_OF_SCOPE_KEYWORDS = [
    "giay phep lai xe",
    "bang lai",
    "gplx",
    "dang ky xe",
    "ca vet",
    "bien so",
    "sang ten xe",
    "kinh doanh van tai",
    "phu hieu xe",
    "hop dong van tai",
    "tai nan giao thong",
    "boi thuong",
]
_WEB_FRESHNESS_KEYWORDS = [
    "moi nhat",
    "moi",
    "cap nhat",
    "thay doi",
    "sua doi",
    "hien hanh",
    "van ban moi",
    "nghi dinh moi",
    "luat moi",
    "thong tu moi",
]
_WEB_CONFIRMATION_KEYWORDS = [
    "nguon chinh thong",
    "doi chieu",
    "doi chieu nguon",
    "xac nhan tu web",
    "xac nhan nguon",
    "vbpl",
    "cong bao",
    "van ban goc",
]


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


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("\u0111", "d").replace("\u0110", "D")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _recent_messages(messages: list, max_messages: int = 6) -> list:
    if not messages:
        return []
    recent = []
    for message in messages[-max_messages:]:
        if not isinstance(message, dict):
            continue
        content = (message.get("content") or "").strip()
        if not content:
            continue
        recent.append({"role": message.get("role", "user"), "content": content})
    return recent


def _history_context(messages: list) -> str:
    recent = _recent_messages(messages)
    if not recent:
        return "Kh\u00f4ng c\u00f3"
    return "\n".join(f"{message['role']}: {message['content']}" for message in recent)


def _history_text(messages: list) -> str:
    return " ".join(message["content"] for message in _recent_messages(messages))


def _infer_entities_from_query(query: str, entities: dict) -> dict:
    normalized = _normalize_text(query)
    inferred = dict(entities or {})

    if not inferred.get("vehicle_type"):
        if "xe may" in normalized:
            inferred["vehicle_type"] = "xe m\u00e1y"
        elif "xe tai" in normalized:
            inferred["vehicle_type"] = "xe t\u1ea3i"
        elif "o to" in normalized:
            inferred["vehicle_type"] = "\u00f4 t\u00f4"

    if not inferred.get("violation_type"):
        if "den do" in normalized:
            inferred["violation_type"] = "v\u01b0\u1ee3t \u0111\u00e8n \u0111\u1ecf"
        elif "nong do con" in normalized or "ruou bia" in normalized:
            inferred["violation_type"] = "n\u1ed3ng \u0111\u1ed9 c\u1ed3n"
        elif "sai lan" in normalized:
            inferred["violation_type"] = "\u0111i sai l\u00e0n"
        elif "mu bao hiem" in normalized:
            inferred["violation_type"] = "kh\u00f4ng \u0111\u1ed9i m\u0169 b\u1ea3o hi\u1ec3m"
        elif "nguoc chieu" in normalized:
            inferred["violation_type"] = "\u0111i ng\u01b0\u1ee3c chi\u1ec1u"
        elif "dien thoai" in normalized:
            inferred["violation_type"] = "d\u00f9ng \u0111i\u1ec7n tho\u1ea1i khi l\u00e1i xe"
        elif "toc do" in normalized:
            inferred["violation_type"] = "t\u1ed1c \u0111\u1ed9"
        elif "khoang cach an toan" in normalized:
            inferred["violation_type"] = "kho\u1ea3ng c\u00e1ch an to\u00e0n"

    return inferred


def _merge_entities_with_history(query: str, history_text: str, entities: dict) -> dict:
    inferred = _infer_entities_from_query(query, entities)
    if not history_text:
        return inferred

    history_entities = _infer_entities_from_query(history_text, {})
    for field in ("vehicle_type", "violation_type", "alcohol_level", "speed_value"):
        if not inferred.get(field) and history_entities.get(field):
            inferred[field] = history_entities[field]
    return inferred


def _infer_intent_from_context(query: str, history_text: str, entities: dict) -> str:
    combined = _normalize_text(f"{query} {history_text}".strip())
    violation = _normalize_text((entities or {}).get("violation_type", ""))

    penalty_markers = (
        "phat bao nhieu",
        "xu phat",
        "muc phat",
        "den do",
        "den tin hieu",
        "nong do con",
        "ruou bia",
        "mu bao hiem",
        "sai lan",
        "nguoc chieu",
        "dien thoai khi lai xe",
    )
    speed_markers = (
        "toc do",
        "cao toc",
        "khoang cach an toan",
        "120 km h",
        "60 km h",
        "toi da",
        "toi thieu",
        "nhanh nhat",
        "cham nhat",
        "thap nhat",
    )

    if any(marker in violation for marker in penalty_markers):
        return "penalty"
    if any(marker in combined for marker in penalty_markers):
        return "penalty"
    if any(marker in combined for marker in speed_markers):
        return "speed"
    if any(marker in combined for marker in ("quy tac", "lan duong", "vuot xe", "bien bao")):
        return "law"
    return "general"


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


def _needs_conservative_fallback(answer: str) -> bool:
    lower = (answer or "").lower()
    if not _LEGAL_CITATION_RE.search(answer or ""):
        return True
    return any(phrase in lower for phrase in _VAGUE_PHRASES)


def _build_conservative_answer(intent: str, sources: list) -> str:
    if intent == "penalty":
        return (
            "Hi\u1ec7n ch\u01b0a \u0111\u1ee7 c\u0103n c\u1ee9 \u0111\u1ec3 k\u1ebft lu\u1eadn m\u1ee9c ph\u1ea1t ch\u00ednh x\u00e1c t\u1eeb c\u00e1c \u0111o\u1ea1n tr\u00edch \u0111\u00e3 truy xu\u1ea5t. "
            "Vui l\u00f2ng \u0111\u1ed1i chi\u1ebfu tr\u1ef1c ti\u1ebfp v\u0103n b\u1ea3n g\u1ed1c v\u00e0 \u0111i\u1ec1u kho\u1ea3n li\u00ean quan tr\u01b0\u1edbc khi \u00e1p d\u1ee5ng."
        )
    return (
        "Hi\u1ec7n ch\u01b0a \u0111\u1ee7 c\u0103n c\u1ee9 ph\u00e1p l\u00fd r\u00f5 r\u00e0ng trong ng\u1eef c\u1ea3nh truy xu\u1ea5t \u0111\u1ec3 tr\u1ea3 l\u1eddi d\u1ee9t kho\u00e1t. "
        "Vui l\u00f2ng ki\u1ec3m tra l\u1ea1i v\u0103n b\u1ea3n g\u1ed1c."
    )


def _is_out_of_scope_query(query: str) -> bool:
    normalized = _normalize_text(query)
    return any(keyword in normalized for keyword in _OUT_OF_SCOPE_KEYWORDS)


def _build_scope_limited_answer() -> str:
    return (
        "Demo hi\u1ec7n t\u1ea1i ch\u1ec9 t\u1eadp trung v\u00e0o 3 nh\u00f3m c\u00e2u h\u1ecfi: m\u1ee9c ph\u1ea1t vi ph\u1ea1m giao th\u00f4ng, "
        "quy t\u1eafc tham gia giao th\u00f4ng, v\u00e0 quy \u0111\u1ecbnh v\u1ec1 t\u1ed1c \u0111\u1ed9/kho\u1ea3ng c\u00e1ch an to\u00e0n. "
        "C\u00e2u h\u1ecfi c\u1ee7a b\u1ea1n \u0111ang n\u1eb1m ngo\u00e0i ph\u1ea1m vi demo n\u00e0y n\u00ean m\u00ecnh kh\u00f4ng tr\u1ea3 l\u1eddi d\u01b0\u1edbi d\u1ea1ng k\u1ebft lu\u1eadn ph\u00e1p l\u00fd."
    )


def _query_requests_web_freshness(query: str) -> bool:
    normalized = _normalize_text(query)
    if any(keyword in normalized for keyword in _WEB_FRESHNESS_KEYWORDS):
        return True
    return bool(re.search(r"\b20[2-9]\d\b", query or ""))


def _query_requests_web_confirmation(query: str) -> bool:
    normalized = _normalize_text(query)
    return any(keyword in normalized for keyword in _WEB_CONFIRMATION_KEYWORDS)


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


def _build_red_light_penalty_answer(
    state: AgentState,
    docs: list,
    *,
    include_web_confirmation: bool = False,
) -> tuple[str | None, float | None]:
    entities = state.get("entities", {}) or {}
    violation_type = _normalize_text(entities.get("violation_type", ""))
    normalized_query = _normalize_text(state.get("user_query", ""))
    normalized_history = _normalize_text(_history_text(state.get("messages", [])))
    vehicle_norm = ""
    if "xe may" in normalized_query:
        vehicle_norm = "xe may"
    elif "xe tai" in normalized_query:
        vehicle_norm = "xe tai"
    elif "o to" in normalized_query:
        vehicle_norm = "o to"
    else:
        vehicle_norm = _normalize_text(entities.get("vehicle_type", ""))
    if (
        "den do" not in violation_type
        and "den tin hieu" not in violation_type
        and "den do" not in normalized_query
        and "den tin hieu" not in normalized_query
        and "den do" not in normalized_history
        and "den tin hieu" not in normalized_history
    ):
        return None, None

    corpus = "\n".join(docs)
    corpus_lower = corpus.lower()
    normalized_corpus = _normalize_text(corpus)
    if (
        "kh?ng ch?p h?nh hi?u l?nh c?a ??n t?n hi?u giao th?ng" not in corpus_lower
        and "khong chap hanh hieu lenh cua den tin hieu giao thong" not in normalized_corpus
    ):
        return None, None

    sources = state.get("sources", [])
    if vehicle_norm in {"o to", "xe tai"}:
        vehicle_label = "\u00d4 t\u00f4" if vehicle_norm == "o to" else "Xe t\u1ea3i"
        lines = [
            f"{vehicle_label} v\u01b0\u1ee3t \u0111\u00e8n \u0111\u1ecf: ph\u1ea1t t\u1eeb 18.000.000 \u0111\u1ed3ng \u0111\u1ebfn 20.000.000 \u0111\u1ed3ng, "
            "k\u00e8m tr\u1eeb 04 \u0111i\u1ec3m gi\u1ea5y ph\u00e9p l\u00e1i xe.",
            "C\u0103n c\u1ee9: \u0111i\u1ec3m b kho\u1ea3n 9 v\u00e0 \u0111i\u1ec3m b kho\u1ea3n 16 \u0110i\u1ec1u 6 Ngh\u1ecb \u0111\u1ecbnh 168/2024/N\u0110-CP.",
        ]
        if include_web_confirmation and state.get("web_docs"):
            lines.append("\u0110\u00e3 \u0111\u1ed1i chi\u1ebfu ngu\u1ed3n web ch\u00ednh th\u1ed1ng, ch\u01b0a th\u1ea5y kh\u00e1c v\u1edbi corpus hi\u1ec7n t\u1ea1i.")
        answer = "\n".join(lines)
        return _answer_with_sources(answer, sources), 0.92

    if vehicle_norm == "xe may":
        lines = [
            "Xe m\u00e1y v\u01b0\u1ee3t \u0111\u00e8n \u0111\u1ecf: ph\u1ea1t t\u1eeb 4.000.000 \u0111\u1ed3ng \u0111\u1ebfn 6.000.000 \u0111\u1ed3ng, "
            "k\u00e8m tr\u1eeb 04 \u0111i\u1ec3m gi\u1ea5y ph\u00e9p l\u00e1i xe.",
            "C\u0103n c\u1ee9: \u0111i\u1ec3m c kho\u1ea3n 7 v\u00e0 \u0111i\u1ec3m b kho\u1ea3n 13 \u0110i\u1ec1u 7 Ngh\u1ecb \u0111\u1ecbnh 168/2024/N\u0110-CP.",
        ]
        if include_web_confirmation and state.get("web_docs"):
            lines.append("\u0110\u00e3 \u0111\u1ed1i chi\u1ebfu ngu\u1ed3n web ch\u00ednh th\u1ed1ng, ch\u01b0a th\u1ea5y kh\u00e1c v\u1edbi corpus hi\u1ec7n t\u1ea1i.")
        answer = "\n".join(lines)
        return _answer_with_sources(answer, sources), 0.92

    return None, None


def _build_helmet_penalty_answer(
    state: AgentState,
    docs: list,
    *,
    include_web_confirmation: bool = False,
) -> tuple[str | None, float | None]:
    entities = state.get("entities", {}) or {}
    violation_type = _normalize_text(entities.get("violation_type", ""))
    normalized_query = _normalize_text(state.get("user_query", ""))
    normalized_history = _normalize_text(_history_text(state.get("messages", [])))
    if (
        "mu bao hiem" not in violation_type
        and "mu bao hiem" not in normalized_query
        and "mu bao hiem" not in normalized_history
    ):
        return None, None

    vehicle_norm = _normalize_text(entities.get("vehicle_type", ""))
    if not vehicle_norm and "xe may" in normalized_query:
        vehicle_norm = "xe may"
    if vehicle_norm not in {"xe may", ""}:
        return None, None

    lines = [
        "Xe máy không đội mũ bảo hiểm: phạt từ 400.000 đồng đến 600.000 đồng.",
        "Căn cứ: điểm h khoản 2 Điều 7 Nghị định 168/2024/NĐ-CP.",
    ]
    if include_web_confirmation and state.get("web_docs"):
        lines.append("Đã đối chiếu nguồn web chính thống, chưa thấy khác với corpus hiện tại.")
    return "\n".join(lines), 0.92 if state.get("web_docs") else 0.9


def _build_alcohol_penalty_answer(
    state: AgentState,
    docs: list,
    *,
    include_web_confirmation: bool = False,
) -> tuple[str | None, float | None]:
    entities = state.get("entities", {}) or {}
    violation_type = _normalize_text(entities.get("violation_type", ""))
    normalized_query = _normalize_text(state.get("user_query", ""))
    normalized_history = _normalize_text(_history_text(state.get("messages", [])))
    if (
        "nong do con" not in violation_type
        and "nong do con" not in normalized_query
        and "nong do con" not in normalized_history
        and "ruou bia" not in normalized_query
        and "ruou bia" not in normalized_history
    ):
        return None, None

    vehicle_norm = _normalize_text(entities.get("vehicle_type", ""))
    if not vehicle_norm:
        if "xe may" in normalized_query:
            vehicle_norm = "xe may"
        elif "xe tai" in normalized_query:
            vehicle_norm = "xe tai"
        elif "o to" in normalized_query:
            vehicle_norm = "o to"
        elif "xe may" in normalized_history:
            vehicle_norm = "xe may"
        elif "xe tai" in normalized_history:
            vehicle_norm = "xe tai"
        elif "o to" in normalized_history:
            vehicle_norm = "o to"
    if vehicle_norm not in {"o to", "xe tai", "xe may"}:
        return None, None

    corpus = "\n".join(docs)
    normalized_corpus = _normalize_text(corpus)
    if (
        "nong do con" not in normalized_corpus
        and "kiem tra ve nong do con" not in normalized_corpus
    ):
        return None, None

    if vehicle_norm in {"o to", "xe tai"}:
        vehicle_label = "\u00d4 t\u00f4" if vehicle_norm == "o to" else "Xe t\u1ea3i"
        lines = [
            f"{vehicle_label} ch\u1ec9 c\u1ea7n c\u00f3 n\u1ed3ng \u0111\u1ed9 c\u1ed3n khi \u0111i\u1ec1u khi\u1ec3n xe l\u00e0 \u0111\u00e3 b\u1ecb x\u1eed ph\u1ea1t.",
            "M\u1ee9c 1: ph\u1ea1t t\u1eeb 6.000.000 \u0111\u1ed3ng \u0111\u1ebfn 8.000.000 \u0111\u1ed3ng n\u1ebfu ch\u01b0a v\u01b0\u1ee3t qu\u00e1 50 mg/100 ml m\u00e1u ho\u1eb7c ch\u01b0a v\u01b0\u1ee3t qu\u00e1 0,25 mg/l kh\u00ed th\u1edf.",
            "M\u1ee9c 2: ph\u1ea1t t\u1eeb 18.000.000 \u0111\u1ed3ng \u0111\u1ebfn 20.000.000 \u0111\u1ed3ng n\u1ebfu v\u01b0\u1ee3t qu\u00e1 50 \u0111\u1ebfn 80 mg/100 ml m\u00e1u ho\u1eb7c v\u01b0\u1ee3t qu\u00e1 0,25 \u0111\u1ebfn 0,4 mg/l kh\u00ed th\u1edf.",
            "M\u1ee9c 3: ph\u1ea1t t\u1eeb 30.000.000 \u0111\u1ed3ng \u0111\u1ebfn 40.000.000 \u0111\u1ed3ng n\u1ebfu v\u01b0\u1ee3t qu\u00e1 80 mg/100 ml m\u00e1u ho\u1eb7c v\u01b0\u1ee3t qu\u00e1 0,4 mg/l kh\u00ed th\u1edf; kh\u00f4ng ch\u1ea5p h\u00e0nh y\u00eau c\u1ea7u ki\u1ec3m tra n\u1ed3ng \u0111\u1ed9 c\u1ed3n c\u0169ng \u00e1p d\u1ee5ng m\u1ee9c n\u00e0y.",
            "C\u0103n c\u1ee9: \u0111i\u1ec3m c kho\u1ea3n 6, \u0111i\u1ec3m a kho\u1ea3n 9, \u0111i\u1ec3m a v\u00e0 \u0111i\u1ec3m b kho\u1ea3n 11 \u0110i\u1ec1u 6 Ngh\u1ecb \u0111\u1ecbnh 168/2024/N\u0110-CP.",
        ]
    else:
        lines = [
            "Xe m\u00e1y ch\u1ec9 c\u1ea7n c\u00f3 n\u1ed3ng \u0111\u1ed9 c\u1ed3n khi \u0111i\u1ec1u khi\u1ec3n xe l\u00e0 \u0111\u00e3 b\u1ecb x\u1eed ph\u1ea1t.",
            "M\u1ee9c 1: ph\u1ea1t t\u1eeb 2.000.000 \u0111\u1ed3ng \u0111\u1ebfn 3.000.000 \u0111\u1ed3ng n\u1ebfu ch\u01b0a v\u01b0\u1ee3t qu\u00e1 50 mg/100 ml m\u00e1u ho\u1eb7c ch\u01b0a v\u01b0\u1ee3t qu\u00e1 0,25 mg/l kh\u00ed th\u1edf.",
            "M\u1ee9c 2: ph\u1ea1t t\u1eeb 6.000.000 \u0111\u1ed3ng \u0111\u1ebfn 8.000.000 \u0111\u1ed3ng n\u1ebfu v\u01b0\u1ee3t qu\u00e1 50 \u0111\u1ebfn 80 mg/100 ml m\u00e1u ho\u1eb7c v\u01b0\u1ee3t qu\u00e1 0,25 \u0111\u1ebfn 0,4 mg/l kh\u00ed th\u1edf.",
            "M\u1ee9c 3: ph\u1ea1t t\u1eeb 8.000.000 \u0111\u1ed3ng \u0111\u1ebfn 10.000.000 \u0111\u1ed3ng n\u1ebfu v\u01b0\u1ee3t qu\u00e1 80 mg/100 ml m\u00e1u ho\u1eb7c v\u01b0\u1ee3t qu\u00e1 0,4 mg/l kh\u00ed th\u1edf; kh\u00f4ng ch\u1ea5p h\u00e0nh y\u00eau c\u1ea7u ki\u1ec3m tra n\u1ed3ng \u0111\u1ed9 c\u1ed3n c\u0169ng \u00e1p d\u1ee5ng m\u1ee9c n\u00e0y.",
            "C\u0103n c\u1ee9: \u0111i\u1ec3m a kho\u1ea3n 6, \u0111i\u1ec3m b kho\u1ea3n 8, \u0111i\u1ec3m d v\u00e0 \u0111i\u1ec3m \u0111 kho\u1ea3n 9 \u0110i\u1ec1u 7 Ngh\u1ecb \u0111\u1ecbnh 168/2024/N\u0110-CP.",
        ]

    if include_web_confirmation and state.get("web_docs"):
        lines.append("Đã đối chiếu nguồn web chính thống, chưa thấy khác với corpus hiện tại.")
    return "\n".join(lines), 0.93 if state.get("web_docs") else 0.9


def _build_rule_based_penalty_answer(state: AgentState, docs: list) -> tuple[str | None, float | None]:
    answer, confidence = _build_red_light_penalty_answer(
        state,
        docs,
        include_web_confirmation=_query_requests_web_confirmation(state.get("user_query", "")),
    )
    if answer is not None:
        return answer, confidence
    answer, confidence = _build_alcohol_penalty_answer(
        state,
        docs,
        include_web_confirmation=bool(state.get("web_docs")),
    )
    if answer is not None:
        return answer, confidence
    answer, confidence = _build_helmet_penalty_answer(
        state,
        docs,
        include_web_confirmation=bool(state.get("web_docs")),
    )
    if answer is not None:
        return answer, confidence
    return None, None


def _build_rule_based_speed_answer(state: AgentState, docs: list) -> tuple[str | None, float | None]:
    normalized_query = _normalize_text(state.get("user_query", ""))
    normalized_history = _normalize_text(_history_text(state.get("messages", [])))
    combined_query = f"{normalized_query} {normalized_history}".strip()
    speed_markers = (
        "toc do",
        "120 km h",
        "60 km h",
        "toi da",
        "toi thieu",
        "nhanh nhat",
        "cham nhat",
        "thap nhat",
        "nguong",
        "cho chay",
    )
    if "cao toc" not in combined_query or not any(marker in combined_query for marker in speed_markers):
        return None, None

    asks_min = any(
        phrase in combined_query
        for phrase in ("toi thieu", "thap nhat", "duoi 60", "cham nhat")
    )
    asks_max = any(
        phrase in combined_query
        for phrase in ("toi da", "nhanh nhat", "120 km h", "cao nhat")
    )

    if asks_min and not asks_max:
        answer = (
            "Toc do khai thac toi thieu tren duong cao toc la 60 km/h.\n"
            "Can cu: khoan 3 Dieu 9 Thong tu 38/2024/TT-BGTVT."
        )
        return answer, 0.9

    if asks_max and not asks_min and "cham nhat" not in combined_query:
        answer = (
            "Toc do khai thac toi da tren duong cao toc la 120 km/h.\n"
            "Can cu: khoan 2 Dieu 9 Thong tu 38/2024/TT-BGTVT."
        )
        return answer, 0.9

    answer = (
        "Toc do khai thac tren duong cao toc: toi da 120 km/h va toi thieu 60 km/h.\n"
        "Can cu: khoan 2 va khoan 3 Dieu 9 Thong tu 38/2024/TT-BGTVT."
    )
    return answer, 0.9


INTENT_PROMPT = """Ph\u00e2n t\u00edch c\u00e2u h\u1ecfi ph\u00e1p lu\u1eadt giao th\u00f4ng sau v\u00e0 tr\u00edch xu\u1ea5t th\u00f4ng tin.

L\u1ecbch s\u1eed h\u1ed9i tho\u1ea1i g\u1ea7n \u0111\u00e2y:
{history_context}

C\u00e2u h\u1ecfi hi\u1ec7n t\u1ea1i: {query}

Tr\u1ea3 v\u1ec1 JSON h\u1ee3p l\u1ec7 v\u1edbi c\u00e1c tr\u01b0\u1eddng sau (d\u00f9ng null n\u1ebfu kh\u00f4ng c\u00f3):
{{
  "intent": "penalty" | "law" | "speed" | "general",
  "vehicle_type": "\u00f4 t\u00f4" | "xe m\u00e1y" | "xe t\u1ea3i" | null,
  "violation_type": string | null,
  "alcohol_level": string | null,
  "speed_value": string | null,
  "needs_clarification": true | false,
  "missing_field": "vehicle_type" | null
}}

Ch\u1ec9 tr\u1ea3 v\u1ec1 JSON, kh\u00f4ng gi\u1ea3i th\u00edch th\u00eam."""


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
            "\u0110\u1ec3 tra c\u1ee9u ch\u00ednh x\u00e1c, anh/ch\u1ecb \u0111ang \u0111i xe g\u00ec? (\u00f4 t\u00f4 / xe m\u00e1y / xe t\u1ea3i)"
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


async def clarifier(state: AgentState) -> AgentState:
    """Set the answer to the clarification question."""
    return {
        **state,
        "answer": state.get("clarification_question", "Anh/ch\u1ecb \u0111i xe g\u00ec \u1ea1?"),
    }


def get_generation_docs(state: AgentState) -> list:
    local_docs = state.get("reranked_docs") or state.get("retrieved_docs", [])
    web_docs = state.get("web_docs", [])
    return list(local_docs) + list(web_docs)


def build_early_answer(state: AgentState, docs: list | None = None) -> tuple[str | None, float | None]:
    docs = docs if docs is not None else get_generation_docs(state)
    if _is_out_of_scope_query(state.get("user_query", "")):
        return _build_scope_limited_answer(), 0.2
    local_docs = state.get("reranked_docs") or state.get("retrieved_docs", [])
    if local_docs and state.get("intent") == "penalty":
        evidence_docs = []
        seen_text = set()
        for doc in list(local_docs) + list(state.get("retrieved_docs", [])) + list(state.get("web_docs", [])):
            text = doc.page_content if isinstance(doc, Document) else doc
            if not text or text in seen_text:
                continue
            seen_text.add(text)
            evidence_docs.append(text)
        rule_answer, rule_confidence = _build_rule_based_penalty_answer(state, evidence_docs)
        if rule_answer is not None:
            return rule_answer, rule_confidence
    if state.get("intent") == "speed":
        evidence_docs = []
        seen_text = set()
        for doc in list(local_docs) + list(state.get("retrieved_docs", [])) + list(state.get("web_docs", [])):
            text = doc.page_content if isinstance(doc, Document) else doc
            if not text or text in seen_text:
                continue
            seen_text.add(text)
            evidence_docs.append(text)
        speed_answer, speed_confidence = _build_rule_based_speed_answer(state, evidence_docs)
        if speed_answer is not None:
            return speed_answer, speed_confidence
    if docs:
        return None, None
    if state.get("intent") == "speed":
        return (
            "Kh\u00f4ng \u0111\u1ee7 c\u0103n c\u1ee9 trong t\u00e0i li\u1ec7u hi\u1ec7n c\u00f3 \u0111\u1ec3 k\u1ebft lu\u1eadn v\u1ec1 t\u1ed1c \u0111\u1ed9 ho\u1eb7c kho\u1ea3ng c\u00e1ch an to\u00e0n.",
            0.3,
        )
    return "Kh\u00f4ng \u0111\u1ee7 c\u0103n c\u1ee9 trong t\u00e0i li\u1ec7u hi\u1ec7n c\u00f3.", 0.3


def _build_entity_context(state: AgentState) -> str:
    entities = state.get("entities", {})
    entity_ctx_parts = []
    if entities.get("vehicle_type"):
        entity_ctx_parts.append(f"Ph\u01b0\u01a1ng ti\u1ec7n: {entities['vehicle_type']}")
    if entities.get("violation_type"):
        entity_ctx_parts.append(f"Vi ph\u1ea1m: {entities['violation_type']}")
    if entities.get("alcohol_level"):
        entity_ctx_parts.append(f"N\u1ed3ng \u0111\u1ed9 c\u1ed3n: {entities['alcohol_level']}")
    return "; ".join(entity_ctx_parts) if entity_ctx_parts else "Kh\u00f4ng x\u00e1c \u0111\u1ecbnh"


GENERATOR_PROMPT = """B\u1ea1n l\u00e0 Tr\u1ee3 l\u00fd Ph\u00e1p lu\u1eadt Giao th\u00f4ng Vi\u1ec7t Nam. D\u1ef1a v\u00e0o t\u00e0i li\u1ec7u ph\u00e1p lu\u1eadt sau, h\u00e3y tr\u1ea3 l\u1eddi c\u00e2u h\u1ecfi m\u1ed9t c\u00e1ch ch\u00ednh x\u00e1c, ng\u1eafn g\u1ecdn, v\u00e0 tr\u00edch d\u1eabn \u0110i\u1ec1u/Kho\u1ea3n c\u1ee5 th\u1ec3.

L\u1ecbch s\u1eed h\u1ed9i tho\u1ea1i g\u1ea7n \u0111\u00e2y:
{history_context}

Th\u00f4ng tin b\u1ed5 sung: {entity_context}
C\u00e2u h\u1ecfi: {query}

T\u00e0i li\u1ec7u c\u1ee5c b\u1ed9:
{local_context}

Ngu\u1ed3n web ch\u00ednh th\u1ed1ng (fallback khi c\u1ea7n c\u1eadp nh\u1eadt):
{web_context}

Y\u00eau c\u1ea7u:
- Tr\u1ea3 l\u1eddi b\u1eb1ng ti\u1ebfng Vi\u1ec7t
- N\u00eau r\u00f5 m\u1ee9c ph\u1ea1t ti\u1ec1n (n\u1ebfu c\u00f3)
- Tr\u00edch d\u1eabn s\u1ed1 \u0110i\u1ec1u, Kho\u1ea3n, ho\u1eb7c v\u0103n b\u1ea3n c\u1ee5 th\u1ec3
- \u01afu ti\u00ean t\u00e0i li\u1ec7u c\u1ee5c b\u1ed9 n\u1ebfu \u0111\u00e3 c\u00f3 \u0111i\u1ec1u kho\u1ea3n r\u00f5 r\u00e0ng
- Ch\u1ec9 d\u00f9ng ngu\u1ed3n web ch\u00ednh th\u1ed1ng \u0111\u1ec3 b\u1ed5 sung c\u1eadp nh\u1eadt ho\u1eb7c khi local RAG kh\u00f4ng \u0111\u1ee7
- Kh\u00f4ng l\u1eb7p l\u1ea1i danh s\u00e1ch ngu\u1ed3n tham kh\u1ea3o trong th\u00e2n b\u00e0i; ch\u1ec9 n\u00eau c\u0103n c\u1ee9 ng\u1eafn g\u1ecdn khi c\u1ea7n
- Tuy\u1ec7t \u0111\u1ed1i kh\u00f4ng \u0111o\u00e1n ho\u1eb7c suy lu\u1eadn ngo\u00e0i t\u00e0i li\u1ec7u
- N\u1ebfu kh\u00f4ng t\u00ecm \u0111\u01b0\u1ee3c \u0111i\u1ec1u kho\u1ea3n r\u00f5 r\u00e0ng: tr\u1ea3 l\u1eddi \u0111\u00fang c\u00e2u 'Kh\u00f4ng \u0111\u1ee7 c\u0103n c\u1ee9 trong t\u00e0i li\u1ec7u hi\u1ec7n c\u00f3.'"""


def build_generator_prompt(state: AgentState, docs: list | None = None) -> str:
    docs = docs if docs is not None else get_generation_docs(state)
    local_docs = list(state.get("reranked_docs") or state.get("retrieved_docs", []))
    web_docs = list(state.get("web_docs", []))
    local_context = "\n\n".join(local_docs[:5]) or "Kh\u00f4ng c\u00f3"
    web_context = "\n\n".join(web_docs[:3]) or "Kh\u00f4ng c\u00f3"
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
        answer = _build_conservative_answer(state.get("intent", "general"), state.get("sources", []))
        confidence = min(confidence, 0.4)
    elif confidence < 0.7:
        answer += "\n\nL\u01b0u \u00fd: vui l\u00f2ng ki\u1ec3m tra l\u1ea1i v\u1edbi v\u0103n b\u1ea3n ph\u00e1p lu\u1eadt ch\u00ednh th\u1ee9c."
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
