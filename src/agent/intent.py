"""Intent parsing and clarification nodes for the VietLegal pipeline."""

from __future__ import annotations

import json

from src.agent.shared import _is_out_of_scope_query, _normalize_text
from src.agent.state import AgentState
from src.llm import invoke_with_fallback

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
    )
    return any(marker in normalized_text for marker in alcohol_markers)


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
        topic = _violation_topic(normalized)
        if topic == "red_light":
            inferred["violation_type"] = "v\u01b0\u1ee3t \u0111\u00e8n \u0111\u1ecf"
        elif topic == "alcohol":
            inferred["violation_type"] = "n\u1ed3ng \u0111\u1ed9 c\u1ed3n"
        elif topic == "wrong_lane":
            inferred["violation_type"] = "\u0111i sai l\u00e0n"
        elif topic == "helmet":
            inferred["violation_type"] = "kh\u00f4ng \u0111\u1ed9i m\u0169 b\u1ea3o hi\u1ec3m"
        elif topic == "wrong_way":
            inferred["violation_type"] = "\u0111i ng\u01b0\u1ee3c chi\u1ec1u"
        elif topic == "phone":
            inferred["violation_type"] = "d\u00f9ng \u0111i\u1ec7n tho\u1ea1i khi l\u00e1i xe"
        elif topic == "speed":
            inferred["violation_type"] = "t\u1ed1c \u0111\u1ed9"
        elif topic == "distance":
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

async def clarifier(state: AgentState) -> AgentState:
    """Set the answer to the clarification question."""
    return {
        **state,
        "answer": state.get("clarification_question", "Anh/ch\u1ecb \u0111i xe g\u00ec \u1ea1?"),
    }
