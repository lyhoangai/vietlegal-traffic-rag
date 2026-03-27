"""Text normalization and lightweight heuristics for the traffic-law agent."""

from __future__ import annotations

import re
import unicodedata

_OUT_OF_SCOPE_KEYWORDS = (
    "giay phep lai xe",
    "bang lai",
    "gplx",
    "dang ky xe",
    "ca vet",
    "bien so",
    "sang ten xe",
    "sang ten",
    "chuyen quyen so huu",
    "doi chu xe",
    "mua lai",
    "kinh doanh van tai",
    "doanh nghiep van tai",
    "kinh doanh taxi",
    "phu hieu xe",
    "phu hieu",
    "hop dong van tai",
    "xe hop dong",
    "xe du lich",
    "tai nan giao thong",
    "gay tai nan",
    "den bao nhieu tien",
    "boi thuong",
)

_WEB_FRESHNESS_KEYWORDS = (
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
)

_WEB_CONFIRMATION_KEYWORDS = (
    "nguon chinh thong",
    "doi chieu",
    "doi chieu nguon",
    "xac nhan tu web",
    "xac nhan nguon",
    "vbpl",
    "cong bao",
    "van ban goc",
)


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = normalized.replace("\u0111", "d").replace("\u0110", "D")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    haystack = f" {normalized_text.strip()} "
    needle = f" {phrase.strip()} "
    return needle in haystack


def _infer_vehicle_from_text(normalized_text: str) -> str | None:
    if not normalized_text:
        return None
    if any(
        _contains_phrase(normalized_text, marker)
        for marker in ("xe may", "xe gan may", "gan may", "xe mo to", "mo to")
    ):
        return "xe máy"
    if any(
        _contains_phrase(normalized_text, marker)
        for marker in ("xe tai", "tai xe xe tai")
    ):
        return "xe tải"
    if any(
        _contains_phrase(normalized_text, marker)
        for marker in ("o to", "xe o to", "xe con", "xe hoi")
    ):
        return "ô tô"
    return None


def _mentions_red_light(normalized_text: str) -> bool:
    return "den do" in normalized_text or "den tin hieu" in normalized_text


def _mentions_alcohol(normalized_text: str) -> bool:
    alcohol_markers = (
        "nong do con",
        "ruou bia",
        "uong ruou",
        "uong bia",
        "say ruou",
        "say bia",
        "co con",
        "nguong con",
    )
    return any(_contains_phrase(normalized_text, marker) for marker in alcohol_markers)


def _violation_topic(normalized_text: str) -> str | None:
    if not normalized_text:
        return None
    if _mentions_red_light(normalized_text):
        return "red_light"
    if _mentions_alcohol(normalized_text):
        return "alcohol"
    if "sai lan" in normalized_text:
        return "wrong_lane"
    if "mu bao hiem" in normalized_text:
        return "helmet"
    if "nguoc chieu" in normalized_text:
        return "wrong_way"
    if "dien thoai" in normalized_text:
        return "phone"
    if "toc do" in normalized_text:
        return "speed"
    if "khoang cach an toan" in normalized_text:
        return "distance"
    return None


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
        return "Khong co"
    return "\n".join(f"{message['role']}: {message['content']}" for message in recent)


def _history_text(messages: list) -> str:
    return " ".join(message["content"] for message in _recent_messages(messages))


def _infer_entities_from_query(query: str, entities: dict) -> dict:
    normalized = _normalize_text(query)
    inferred = dict(entities or {})

    if not inferred.get("vehicle_type"):
        inferred["vehicle_type"] = _infer_vehicle_from_text(normalized)

    if not inferred.get("violation_type"):
        topic = _violation_topic(normalized)
        if topic == "red_light":
            inferred["violation_type"] = "vượt đèn đỏ"
        elif topic == "alcohol":
            inferred["violation_type"] = "nồng độ cồn"
        elif topic == "wrong_lane":
            inferred["violation_type"] = "đi sai làn"
        elif topic == "helmet":
            inferred["violation_type"] = "không đội mũ bảo hiểm"
        elif topic == "wrong_way":
            inferred["violation_type"] = "đi ngược chiều"
        elif topic == "phone":
            inferred["violation_type"] = "dùng điện thoại khi lái xe"
        elif topic == "speed":
            inferred["violation_type"] = "tốc độ"
        elif topic == "distance":
            inferred["violation_type"] = "khoảng cách an toàn"

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
        "co con",
        "nguong con",
        "ruou bia",
        "uong ruou",
        "uong bia",
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


def _is_out_of_scope_query(query: str) -> bool:
    normalized = _normalize_text(query)
    return any(keyword in normalized for keyword in _OUT_OF_SCOPE_KEYWORDS)


def _query_requests_web_freshness(query: str) -> bool:
    normalized = _normalize_text(query)
    if any(keyword in normalized for keyword in _WEB_FRESHNESS_KEYWORDS):
        return True
    return bool(re.search(r"\b20[2-9]\d\b", query or ""))


def _query_requests_web_confirmation(query: str) -> bool:
    normalized = _normalize_text(query)
    return any(keyword in normalized for keyword in _WEB_CONFIRMATION_KEYWORDS)
