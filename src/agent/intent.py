"""Intent parsing and clarification nodes for the VietLegal pipeline."""

from __future__ import annotations

import json

from src.agent.state import AgentState
from src.agent.text_utils import (
    _history_context,
    _history_text,
    _infer_entities_from_query,
    _infer_intent_from_context,
    _is_out_of_scope_query,
    _merge_entities_with_history,
    _violation_topic,
)
from src.llm import invoke_with_fallback


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


def _parse_intent_json(raw: str) -> dict:
    clean = (raw or "").strip()
    if clean.startswith("```json"):
        clean = clean[len("```json") :].strip()
    elif clean.startswith("```"):
        clean = clean[len("```") :].strip()
    if clean.endswith("```"):
        clean = clean[:-3].strip()
    try:
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        return {"intent": "general", "needs_clarification": False}


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

    extracted = _parse_intent_json(raw)
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


async def clarifier(state: AgentState) -> AgentState:
    """Set the answer to the clarification question."""
    return {
        **state,
        "answer": state.get("clarification_question", "Anh/chị đi xe gì ạ?"),
    }
