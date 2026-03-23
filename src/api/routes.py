"""Chat API routes for non-streaming, streaming, history, and TTS responses."""

from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import Response
from sse_starlette.sse import EventSourceResponse

from src.agent.graph import agent
from src.agent.nodes import (
    build_early_answer,
    build_generator_prompt,
    clarifier,
    finalize_generated_answer,
    get_generation_docs,
    intent_analyzer,
    query_router,
    reranker,
    retriever,
    web_searcher,
)
from src.eval.db import get_avg_metrics
from src.llm import stream_with_fallback
from src.memory.store import (
    delete_session_messages,
    get_recent_sessions,
    get_session_history,
    get_session_messages,
    remember_turn,
)
from src.tts.edge_tts_wrapper import (
    TTSUnavailableError,
    list_edge_voices,
    synthesize_edge_tts_bytes,
)

router = APIRouter()
_MAX_SESSION_MESSAGES = 12


class ChatRequest(BaseModel):
    query: str
    session_id: str = ""


class TTSRequest(BaseModel):
    text: str
    voice: str = os.getenv("EDGE_TTS_VOICE", "vi-VN-HoaiMyNeural")
    rate: str = os.getenv("EDGE_TTS_RATE", "+0%")


def _web_used(result: dict) -> bool:
    if result.get("web_docs"):
        return True
    return any(str(source).startswith("Web ") for source in result.get("sources", []))


def _make_state(query: str, session_id: str = "") -> dict:
    return {
        "messages": get_session_messages(session_id, limit=_MAX_SESSION_MESSAGES),
        "user_query": query,
        "intent": "general",
        "entities": {},
        "retrieved_docs": [],
        "reranked_docs": [],
        "web_docs": [],
        "sources": [],
        "needs_clarification": False,
        "clarification_question": "",
        "answer": "",
        "confidence": 0.0,
        "llm_provider": os.getenv("LLM_PROVIDER", "gemini"),
        "collection_used": "traffic_law",
    }


def _response_payload(result: dict) -> dict:
    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "sources": result.get("sources", []),
        "web_used": _web_used(result),
        "llm_provider": result["llm_provider"],
        "collection_used": result.get("collection_used", ""),
        "needs_clarification": result.get("needs_clarification", False),
        "intent": result.get("intent", "general"),
        "entities": result.get("entities", {}),
    }


def _done_event_payload(result: dict) -> dict:
    return {
        "type": "done",
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "web_used": _web_used(result),
        "confidence": result.get("confidence", 0.0),
        "llm_provider": result.get("llm_provider", "gemini"),
        "collection_used": result.get("collection_used", ""),
        "intent": result.get("intent", "general"),
        "entities": result.get("entities", {}),
        "needs_clarification": result.get("needs_clarification", False),
    }


@router.get("/chat/history")
async def chat_history(session_id: str = ""):
    """Return recent persisted messages for a session."""
    return get_session_history(session_id, limit=_MAX_SESSION_MESSAGES)


@router.get("/chat/sessions")
async def chat_sessions(limit: int = 20):
    """Return recent chat sessions for the sidebar."""
    safe_limit = max(1, min(limit, 50))
    return {"sessions": get_recent_sessions(limit=safe_limit)}


@router.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete all messages for a session."""
    delete_session_messages(session_id)
    return {"deleted": True, "session_id": session_id}


@router.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming endpoint. Runs the full graph and returns the final result."""
    state = _make_state(request.query, request.session_id)
    result = await agent.ainvoke(state)
    remember_turn(request.session_id, request.query, result.get("answer", ""))
    return _response_payload(result)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE streaming endpoint with real provider token streaming."""
    state = _make_state(request.query, request.session_id)

    async def event_gen():
        current_state = await intent_analyzer(state)
        yield {
            "data": json.dumps(
                {
                    "type": "node",
                    "node": "intent_analyzer",
                    "intent": current_state.get("intent"),
                    "entities": current_state.get("entities"),
                    "collection": current_state.get("collection_used"),
                }
            )
        }

        if current_state.get("needs_clarification"):
            current_state = await clarifier(current_state)
            answer = current_state.get("answer", "")
            if answer:
                yield {"data": json.dumps({"type": "text", "content": answer})}
            remember_turn(request.session_id, request.query, answer)
            yield {"data": json.dumps(_done_event_payload(current_state))}
            return

        current_state = await query_router(current_state)
        current_state = await retriever(current_state)
        current_state = await reranker(current_state)
        yield {
            "data": json.dumps(
                {
                    "type": "node",
                    "node": "retriever",
                    "intent": current_state.get("intent"),
                    "entities": current_state.get("entities"),
                    "collection": current_state.get("collection_used"),
                    "docs_count": len(current_state.get("reranked_docs", [])),
                }
            )
        }

        yield {"data": json.dumps({"type": "node", "node": "web_searcher"})}
        current_state = await web_searcher(current_state)

        yield {"data": json.dumps({"type": "node", "node": "generator"})}

        docs = get_generation_docs(current_state)
        early_answer, early_confidence = build_early_answer(current_state, docs)
        if early_answer is not None:
            current_state = {
                **current_state,
                "answer": early_answer,
                "confidence": early_confidence,
            }
            yield {"data": json.dumps({"type": "text", "content": early_answer})}
            remember_turn(request.session_id, request.query, early_answer)
            yield {"data": json.dumps(_done_event_payload(current_state))}
            return

        prompt = build_generator_prompt(current_state, docs)
        chunks = []
        async for token in stream_with_fallback(prompt, current_state):
            chunks.append(token)
            yield {"data": json.dumps({"type": "text", "content": token})}

        raw_answer = "".join(chunks).strip()
        final_answer, confidence = finalize_generated_answer(current_state, raw_answer, docs)
        current_state = {
            **current_state,
            "answer": final_answer,
            "confidence": confidence,
        }
        if final_answer != raw_answer:
            yield {"data": json.dumps({"type": "replace", "content": final_answer})}

        remember_turn(request.session_id, request.query, final_answer)
        yield {"data": json.dumps(_done_event_payload(current_state))}

    return EventSourceResponse(event_gen())


@router.get("/eval/metrics")
async def eval_metrics():
    """Return average Ragas evaluation metrics from SQLite."""
    return get_avg_metrics()


@router.get("/tts/voices")
async def tts_voices(locale: str = "vi-VN"):
    """Return the available edge-tts voices for a locale."""
    try:
        return {"voices": await list_edge_voices(locale=locale)}
    except TTSUnavailableError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=503, detail=f"TTS voice lookup failed: {err}") from err


@router.post("/tts")
async def tts_synthesize(request: TTSRequest):
    """Synthesize an answer with edge-tts and return MP3 bytes."""
    try:
        audio_bytes = await synthesize_edge_tts_bytes(
            request.text,
            voice=request.voice,
            rate=request.rate,
        )
    except TTSUnavailableError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=503, detail=f"TTS synthesis failed: {err}") from err

    headers = {"x-tts-voice": request.voice, "cache-control": "no-store"}
    return Response(content=audio_bytes, media_type="audio/mpeg", headers=headers)
