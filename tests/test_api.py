"""tests/test_api.py - FastAPI endpoint tests."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
import src.api.routes as routes


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "VietLegal Traffic RAG"


@pytest.mark.asyncio
async def test_index_serves_dark_chat_ui():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "VietLegal Traffic RAG" in response.text
    assert "/chat/stream" in response.text
    assert "/tts" in response.text
    assert "voice-select" in response.text


@pytest.mark.asyncio
async def test_eval_metrics_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/eval/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "context_precision" in data
    assert "faithfulness" in data
    assert "answer_correctness" in data


@pytest.mark.asyncio
async def test_chat_sessions_endpoint_returns_recent_sessions(monkeypatch):
    monkeypatch.setattr(
        routes,
        "get_recent_sessions",
        lambda limit=20: [
            {
                "session_id": "s1",
                "title": "Ô tô vượt đèn đỏ phạt bao nhiêu",
                "last_message": "Phạt từ 18 triệu đến 20 triệu",
                "updated_at": 1710000000.0,
                "message_count": 2,
            }
        ],
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/chat/sessions?limit=10")

    assert response.status_code == 200
    data = response.json()
    assert data["sessions"][0]["session_id"] == "s1"
    assert "Ô tô vượt đèn đỏ" in data["sessions"][0]["title"]


@pytest.mark.asyncio
async def test_delete_chat_session_endpoint(monkeypatch):
    deleted = {}

    def fake_delete_session_messages(session_id: str):
        deleted["session_id"] = session_id

    monkeypatch.setattr(routes, "delete_session_messages", fake_delete_session_messages)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete("/chat/sessions/session-123")

    assert response.status_code == 200
    assert response.json() == {"deleted": True, "session_id": "session-123"}
    assert deleted["session_id"] == "session-123"


@pytest.mark.asyncio
async def test_tts_voices_endpoint_returns_backend_voices(monkeypatch):
    async def fake_list_edge_voices(locale: str = "vi-VN"):
        assert locale == "vi-VN"
        return [
            {
                "short_name": "vi-VN-HoaiMyNeural",
                "display_name": "Microsoft HoaiMy Online (Natural) - Vietnamese (Vietnam)",
                "locale": "vi-VN",
                "gender": "Female",
            }
        ]

    monkeypatch.setattr(routes, "list_edge_voices", fake_list_edge_voices)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/tts/voices?locale=vi-VN")

    assert response.status_code == 200
    data = response.json()
    assert data["voices"][0]["short_name"] == "vi-VN-HoaiMyNeural"


@pytest.mark.asyncio
async def test_tts_endpoint_returns_audio_bytes(monkeypatch):
    async def fake_synthesize_edge_tts_bytes(text: str, voice: str | None = None, rate: str | None = None):
        assert text == "Xin chao"
        assert voice == "vi-VN-HoaiMyNeural"
        assert rate == "+0%"
        return b"ID3\x01\x02\x03"

    monkeypatch.setattr(routes, "synthesize_edge_tts_bytes", fake_synthesize_edge_tts_bytes)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/tts",
            json={"text": "Xin chao", "voice": "vi-VN-HoaiMyNeural", "rate": "+0%"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.headers["x-tts-voice"] == "vi-VN-HoaiMyNeural"
    assert response.content == b"ID3\x01\x02\x03"


@pytest.mark.asyncio
async def test_chat_stream_endpoint_emits_done_with_answer(monkeypatch):
    async def fake_stream_chat_turn(state: dict):
        yield {
            "type": "node",
            "node": "intent_analyzer",
            "intent": "penalty",
            "entities": {"vehicle_type": "o to"},
            "collection": "traffic_penalties",
        }
        yield {
            "type": "node",
            "node": "retriever",
            "intent": "penalty",
            "entities": {"vehicle_type": "o to"},
            "collection": "traffic_penalties",
            "docs_count": 2,
        }
        yield {"type": "node", "node": "web_searcher"}
        yield {"type": "text", "content": "M"}
        yield {
            "type": "done",
            "state": {
                **state,
                "answer": "Muc phat",
                "confidence": 0.9,
                "web_docs": ["web_doc"],
                "sources": ["source_a", "web_source"],
            },
        }

    monkeypatch.setattr(routes, "stream_chat_turn", fake_stream_chat_turn)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=30.0) as client:
        async with client.stream("POST", "/chat/stream", json={"query": "test query", "session_id": "s2"}) as response:
            body = ""
            async for chunk in response.aiter_text():
                body += chunk

    assert response.status_code == 200
    assert '"type": "node"' in body
    assert '"node": "intent_analyzer"' in body
    assert '"node": "retriever"' in body
    assert '"node": "web_searcher"' in body
    assert '"content": "M"' in body
    assert '"type": "done"' in body
    assert '"answer": "Muc phat"' in body
    assert '"web_used": true' in body


@pytest.mark.asyncio
async def test_chat_endpoint_includes_web_used_when_result_has_web_docs(monkeypatch):
    async def fake_run_chat_turn(state: dict):
        return {
            **state,
            "answer": "Cap nhat tu nguon web",
            "confidence": 0.8,
            "sources": ["Web · vbpl.vn · Nghi dinh 168/2024/NĐ-CP"],
            "web_docs": ["[Nguồn web chính thống]"],
            "intent": "penalty",
            "llm_provider": "groq",
        }

    monkeypatch.setattr(routes, "run_chat_turn", fake_run_chat_turn)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/chat", json={"query": "test query", "session_id": "s3"})

    assert response.status_code == 200
    data = response.json()
    assert data["web_used"] is True
