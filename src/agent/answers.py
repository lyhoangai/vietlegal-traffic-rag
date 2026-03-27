"""Answer-building helpers and generation node for the VietLegal pipeline."""

from __future__ import annotations

from langchain_core.documents import Document

from src.agent.rule_based import (
    _build_conservative_answer,
    _build_rule_based_penalty_answer,
    _build_rule_based_speed_answer,
    _build_scope_limited_answer,
    _needs_conservative_fallback,
)
from src.agent.state import AgentState
from src.agent.text_utils import _history_context, _is_out_of_scope_query
from src.llm import invoke_with_fallback


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
