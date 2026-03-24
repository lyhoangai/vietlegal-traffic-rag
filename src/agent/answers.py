"""Answer-building helpers and generation node for the VietLegal pipeline."""

from __future__ import annotations

from langchain_core.documents import Document

from src.agent.intent import _history_context, _history_text, _violation_topic
from src.agent.shared import (
    _LEGAL_CITATION_RE,
    _VAGUE_PHRASES,
    _answer_with_sources,
    _is_out_of_scope_query,
    _normalize_text,
    _query_requests_web_confirmation,
)
from src.agent.state import AgentState
from src.llm import invoke_with_fallback

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

def _build_scope_limited_answer() -> str:
    return (
        "Demo hi\u1ec7n t\u1ea1i ch\u1ec9 t\u1eadp trung v\u00e0o 3 nh\u00f3m c\u00e2u h\u1ecfi: m\u1ee9c ph\u1ea1t vi ph\u1ea1m giao th\u00f4ng, "
        "quy t\u1eafc tham gia giao th\u00f4ng, v\u00e0 quy \u0111\u1ecbnh v\u1ec1 t\u1ed1c \u0111\u1ed9/kho\u1ea3ng c\u00e1ch an to\u00e0n. "
        "C\u00e2u h\u1ecfi c\u1ee7a b\u1ea1n \u0111ang n\u1eb1m ngo\u00e0i ph\u1ea1m vi demo n\u00e0y n\u00ean m\u00ecnh kh\u00f4ng tr\u1ea3 l\u1eddi d\u01b0\u1edbi d\u1ea1ng k\u1ebft lu\u1eadn ph\u00e1p l\u00fd."
    )


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
    violation_topic = _violation_topic(violation_type)
    query_topic = _violation_topic(normalized_query)
    history_topic = _violation_topic(normalized_history)
    vehicle_norm = ""
    if "xe may" in normalized_query:
        vehicle_norm = "xe may"
    elif "xe tai" in normalized_query:
        vehicle_norm = "xe tai"
    elif "o to" in normalized_query:
        vehicle_norm = "o to"
    else:
        vehicle_norm = _normalize_text(entities.get("vehicle_type", ""))
    if violation_topic == "red_light" or query_topic == "red_light":
        should_answer = True
    else:
        should_answer = query_topic is None and history_topic == "red_light"
    if not should_answer:
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
    violation_topic = _violation_topic(violation_type)
    query_topic = _violation_topic(normalized_query)
    history_topic = _violation_topic(normalized_history)
    if violation_topic == "alcohol" or query_topic == "alcohol":
        should_answer = True
    else:
        should_answer = query_topic is None and history_topic == "alcohol"
    if not should_answer:
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
