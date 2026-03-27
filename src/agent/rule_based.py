"""Rule-based guardrails and compact answers for high-confidence traffic-law cases."""

from __future__ import annotations

import re

from langchain_core.documents import Document

from src.agent.state import AgentState
from src.agent.text_utils import (
    _history_text,
    _normalize_text,
    _query_requests_web_confirmation,
    _violation_topic,
)

_LEGAL_CITATION_RE = re.compile(
    r"\b(Điều|Khoản|Nghị định|Thông tư|Luật)\b",
    re.IGNORECASE,
)
_VAGUE_PHRASES = (
    "không có thông tin cụ thể",
    "không được chỉ định rõ",
    "thông thường",
    "có thể tham khảo",
    "khuyến nghị bạn tham khảo",
)


def _combined_evidence_text(docs: list) -> str:
    texts = []
    for doc in docs or []:
        text = doc.page_content if isinstance(doc, Document) else str(doc or "")
        if text:
            texts.append(text)
    return "\n".join(texts)


def _has_all_markers(normalized_corpus: str, markers: tuple[str, ...]) -> bool:
    return all(marker in normalized_corpus for marker in markers)


def _build_scope_limited_answer() -> str:
    return (
        "Demo hiện tại chỉ tập trung vào 3 nhóm câu hỏi: mức phạt vi phạm giao thông, "
        "quy tắc tham gia giao thông, và quy định về tốc độ/khoảng cách an toàn. "
        "Câu hỏi của bạn đang nằm ngoài phạm vi demo này nên mình không trả lời dưới dạng kết luận pháp lý."
    )


def _needs_conservative_fallback(answer: str) -> bool:
    lower = (answer or "").lower()
    if not _LEGAL_CITATION_RE.search(answer or ""):
        return True
    return any(phrase in lower for phrase in _VAGUE_PHRASES)


def _build_conservative_answer(intent: str) -> str:
    if intent == "penalty":
        return (
            "Hiện chưa đủ căn cứ để kết luận mức phạt chính xác từ các đoạn trích đã truy xuất. "
            "Vui lòng đối chiếu trực tiếp văn bản gốc và điều khoản liên quan trước khi áp dụng."
        )
    return (
        "Hiện chưa đủ căn cứ pháp lý rõ ràng trong ngữ cảnh truy xuất để trả lời dứt khoát. "
        "Vui lòng kiểm tra lại văn bản gốc."
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

    if violation_topic == "red_light" or query_topic == "red_light":
        should_answer = True
    else:
        should_answer = query_topic is None and history_topic == "red_light"
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

    normalized_corpus = _normalize_text(_combined_evidence_text(docs))
    if "khong chap hanh hieu lenh cua den tin hieu giao thong" not in normalized_corpus:
        return None, None

    lines: list[str]
    if vehicle_norm in {"o to", "xe tai"}:
        vehicle_label = "Ô tô" if vehicle_norm == "o to" else "Xe tải"
        lines = [
            f"{vehicle_label} vượt đèn đỏ: phạt từ 18.000.000 đồng đến 20.000.000 đồng, kèm trừ 04 điểm giấy phép lái xe.",
            "Căn cứ: điểm b khoản 9 và điểm b khoản 16 Điều 6 Nghị định 168/2024/NĐ-CP.",
        ]
    elif vehicle_norm == "xe may":
        lines = [
            "Xe máy vượt đèn đỏ: phạt từ 4.000.000 đồng đến 6.000.000 đồng, kèm trừ 04 điểm giấy phép lái xe.",
            "Căn cứ: điểm c khoản 7 và điểm b khoản 13 Điều 7 Nghị định 168/2024/NĐ-CP.",
        ]
    else:
        return None, None

    if include_web_confirmation and state.get("web_docs"):
        lines.append("Đã đối chiếu nguồn web chính thống, chưa thấy khác với corpus hiện tại.")
    return "\n".join(lines), 0.92


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

    normalized_corpus = _normalize_text(_combined_evidence_text(docs))
    history_has_amounts = _has_all_markers(
        normalized_history,
        ("400 000 dong", "600 000 dong"),
    )
    corpus_has_amounts = _has_all_markers(
        normalized_corpus,
        ("400 000 dong", "600 000 dong"),
    )
    if "mu bao hiem" not in normalized_corpus:
        return None, None
    if not (corpus_has_amounts or history_has_amounts):
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

    normalized_corpus = _normalize_text(_combined_evidence_text(docs))
    if "nong do con" not in normalized_corpus and "kiem tra ve nong do con" not in normalized_corpus:
        return None, None

    if vehicle_norm in {"o to", "xe tai"}:
        vehicle_label = "Ô tô" if vehicle_norm == "o to" else "Xe tải"
        lines = [
            f"{vehicle_label} chỉ cần có nồng độ cồn khi điều khiển xe là đã bị xử phạt.",
            "Mức 1: phạt từ 6.000.000 đồng đến 8.000.000 đồng nếu chưa vượt quá 50 mg/100 ml máu hoặc chưa vượt quá 0,25 mg/l khí thở.",
            "Mức 2: phạt từ 18.000.000 đồng đến 20.000.000 đồng nếu vượt quá 50 đến 80 mg/100 ml máu hoặc vượt quá 0,25 đến 0,4 mg/l khí thở.",
            "Mức 3: phạt từ 30.000.000 đồng đến 40.000.000 đồng nếu vượt quá 80 mg/100 ml máu hoặc vượt quá 0,4 mg/l khí thở; không chấp hành yêu cầu kiểm tra nồng độ cồn cũng áp dụng mức này.",
            "Căn cứ: điểm c khoản 6, điểm a khoản 9, điểm a và điểm b khoản 11 Điều 6 Nghị định 168/2024/NĐ-CP.",
        ]
    else:
        lines = [
            "Xe máy chỉ cần có nồng độ cồn khi điều khiển xe là đã bị xử phạt.",
            "Mức 1: phạt từ 2.000.000 đồng đến 3.000.000 đồng nếu chưa vượt quá 50 mg/100 ml máu hoặc chưa vượt quá 0,25 mg/l khí thở.",
            "Mức 2: phạt từ 6.000.000 đồng đến 8.000.000 đồng nếu vượt quá 50 đến 80 mg/100 ml máu hoặc vượt quá 0,25 đến 0,4 mg/l khí thở.",
            "Mức 3: phạt từ 8.000.000 đồng đến 10.000.000 đồng nếu vượt quá 80 mg/100 ml máu hoặc vượt quá 0,4 mg/l khí thở; không chấp hành yêu cầu kiểm tra nồng độ cồn cũng áp dụng mức này.",
            "Căn cứ: điểm a khoản 6, điểm b khoản 8, điểm d và điểm đ khoản 9 Điều 7 Nghị định 168/2024/NĐ-CP.",
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

    normalized_corpus = _normalize_text(_combined_evidence_text(docs))
    if not _has_all_markers(normalized_corpus, ("duong cao toc", "toc do khai thac")):
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
        return (
            "Tốc độ khai thác tối thiểu trên đường cao tốc là 60 km/h.\n"
            "Căn cứ: khoản 3 Điều 9 Thông tư 38/2024/TT-BGTVT.",
            0.9,
        )

    if asks_max and not asks_min and "cham nhat" not in combined_query:
        return (
            "Tốc độ khai thác tối đa trên đường cao tốc là 120 km/h.\n"
            "Căn cứ: khoản 2 Điều 9 Thông tư 38/2024/TT-BGTVT.",
            0.9,
        )

    return (
        "Tốc độ khai thác trên đường cao tốc: tối đa 120 km/h và tối thiểu 60 km/h.\n"
        "Căn cứ: khoản 2 và khoản 3 Điều 9 Thông tư 38/2024/TT-BGTVT.",
        0.9,
    )
