"""Shared chat orchestration helpers for both streaming and non-streaming API paths."""

from __future__ import annotations

from src.agent.nodes import (
    build_early_answer,
    build_generator_prompt,
    clarifier,
    finalize_generated_answer,
    generator,
    get_generation_docs,
    intent_analyzer,
    query_router,
    reranker,
    retriever,
    web_searcher,
)
from src.agent.state import AgentState
from src.llm import stream_with_fallback


def _intent_node_event(state: AgentState) -> dict:
    return {
        "type": "node",
        "node": "intent_analyzer",
        "intent": state.get("intent"),
        "entities": state.get("entities"),
        "collection": state.get("collection_used"),
    }


def _retriever_node_event(state: AgentState) -> dict:
    return {
        "type": "node",
        "node": "retriever",
        "intent": state.get("intent"),
        "entities": state.get("entities"),
        "collection": state.get("collection_used"),
        "docs_count": len(state.get("reranked_docs", [])),
    }


async def _advance_to_generation(state: AgentState) -> tuple[AgentState, list[dict], bool]:
    events: list[dict] = []

    current_state = await intent_analyzer(state)
    events.append(_intent_node_event(current_state))

    if current_state.get("needs_clarification"):
        current_state = await clarifier(current_state)
        answer = current_state.get("answer", "")
        if answer:
            events.append({"type": "text", "content": answer})
        return current_state, events, True

    current_state = await query_router(current_state)
    current_state = await retriever(current_state)
    current_state = await reranker(current_state)
    events.append(_retriever_node_event(current_state))

    events.append({"type": "node", "node": "web_searcher"})
    current_state = await web_searcher(current_state)
    events.append({"type": "node", "node": "generator"})
    return current_state, events, False


async def run_chat_turn(state: AgentState) -> AgentState:
    current_state, _, done = await _advance_to_generation(state)
    if done:
        return current_state
    return await generator(current_state)


async def stream_chat_turn(state: AgentState):
    current_state, events, done = await _advance_to_generation(state)
    for event in events:
        yield event

    if done:
        yield {"type": "done", "state": current_state}
        return

    docs = get_generation_docs(current_state)
    early_answer, early_confidence = build_early_answer(current_state, docs)
    if early_answer is not None:
        current_state = {
            **current_state,
            "answer": early_answer,
            "confidence": early_confidence,
        }
        yield {"type": "text", "content": early_answer}
        yield {"type": "done", "state": current_state}
        return

    prompt = build_generator_prompt(current_state, docs)
    chunks: list[str] = []
    async for token in stream_with_fallback(prompt, current_state):
        chunks.append(token)
        yield {"type": "text", "content": token}

    raw_answer = "".join(chunks).strip()
    final_answer, confidence = finalize_generated_answer(current_state, raw_answer, docs)
    current_state = {
        **current_state,
        "answer": final_answer,
        "confidence": confidence,
    }
    if final_answer != raw_answer:
        yield {"type": "replace", "content": final_answer}

    yield {"type": "done", "state": current_state}
