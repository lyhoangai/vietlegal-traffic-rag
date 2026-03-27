from unittest.mock import AsyncMock, patch

import pytest

from tests.agent.support import _make_state

@pytest.mark.asyncio
async def test_query_router_penalty():
    from src.agent.retrieval import query_router

    state = _make_state("ô tô vượt đèn đỏ phạt bao nhiêu")
    state["intent"] = "penalty"
    result = await query_router(state)
    assert result["collection_used"] == "traffic_penalties"


@pytest.mark.asyncio
async def test_query_router_speed():
    from src.agent.retrieval import query_router

    state = _make_state("tốc độ tối đa trên đường cao tốc")
    state["intent"] = "speed"
    result = await query_router(state)
    assert result["collection_used"] == "traffic_speed"

async def test_reranker_reduces_to_5():
    from src.agent.retrieval import reranker

    state = _make_state("test")
    state["retrieved_docs"] = [f"doc {i}" for i in range(20)]
    with patch("src.agent.retrieval._reranker_model", None):
        result = await reranker(state)
    assert len(result["reranked_docs"]) == 5


@pytest.mark.asyncio
async def test_reranker_can_be_disabled_via_env(monkeypatch):
    from src.agent.retrieval import reranker

    state = _make_state("test")
    state["retrieved_docs"] = [f"doc {i}" for i in range(8)]
    monkeypatch.setenv("ENABLE_RERANKER", "false")

    with patch("src.agent.retrieval._get_reranker", side_effect=AssertionError("should not load")):
        result = await reranker(state)

    assert result["reranked_docs"] == state["retrieved_docs"][:5]

@pytest.mark.asyncio
async def test_web_searcher_skips_when_local_docs_are_strong():
    from src.agent.retrieval import web_searcher

    state = _make_state("quy tac vuot xe tren duong bo")
    state["intent"] = "law"
    state["reranked_docs"] = ["doc 1", "doc 2", "doc 3"]

    with patch("src.agent.retrieval.search_official_web", new=AsyncMock(return_value=[])) as mock_search:
        result = await web_searcher(state)

    mock_search.assert_not_awaited()
    assert result["web_docs"] == []


@pytest.mark.asyncio
async def test_web_searcher_uses_web_when_user_explicitly_requests_official_source():
    from src.agent.retrieval import web_searcher

    state = _make_state("doi chieu nguon chinh thong quy tac vuot xe")
    state["intent"] = "law"
    state["reranked_docs"] = ["doc 1", "doc 2", "doc 3"]

    with patch(
        "src.agent.retrieval.search_official_web",
        new=AsyncMock(
            return_value=[
                {
                    "title": "Luat 36/2024/QH15",
                    "url": "https://vbpl.vn/Pages/vbpq-toanvan.aspx?ItemID=123456",
                    "content": "Quy tắc vượt xe trên đường bộ.",
                    "source": "Web | vbpl.vn | Luat 36/2024/QH15",
                }
            ]
        ),
    ) as mock_search:
        result = await web_searcher(state)

    mock_search.assert_awaited_once()
    assert result["web_docs"]
    assert any("vbpl.vn" in source for source in result["sources"])


@pytest.mark.asyncio
async def test_web_searcher_merges_official_results_when_local_docs_are_missing():
    from src.agent.retrieval import web_searcher

    state = _make_state("nghi dinh moi ve toc do toi da")
    state["intent"] = "speed"

    with patch(
        "src.agent.retrieval.search_official_web",
        new=AsyncMock(
            return_value=[
                {
                    "title": "Nghi dinh moi",
                    "url": "https://congbao.chinhphu.vn/van-ban/nghi-dinh-moi.htm",
                    "content": "Van ban moi nhat cap nhat toc do toi da.",
                    "source": "Web | congbao.chinhphu.vn | Nghi dinh moi",
                }
            ]
        ),
    ):
        result = await web_searcher(state)

    assert len(result["web_docs"]) == 1
    assert "URL:" in result["web_docs"][0]
    assert "congbao.chinhphu.vn" in result["sources"][-1]


@pytest.mark.asyncio
async def test_web_searcher_uses_web_for_penalty_when_no_rule_based_answer_exists():
    from src.agent.retrieval import web_searcher

    state = _make_state("xe may khong doi mu bao hiem bi xu phat the nao")
    state["intent"] = "penalty"
    state["entities"] = {"vehicle_type": "xe máy", "violation_type": "không đội mũ bảo hiểm"}
    state["reranked_docs"] = ["Điểm b khoản 6 ... không đội mũ bảo hiểm ..."]

    with patch(
        "src.agent.retrieval.search_official_web",
        new=AsyncMock(
            return_value=[
                {
                    "title": "Nghi dinh 168/2024/ND-CP",
                    "url": "https://vbpl.vn/bocongan/Pages/vbpq-toanvan.aspx?ItemID=173920",
                    "content": "Quy định xử phạt vi phạm hành chính về trật tự, an toàn giao thông.",
                    "source": "Web | vbpl.vn | Nghi dinh 168/2024/ND-CP",
                }
            ]
        ),
    ) as mock_search:
        result = await web_searcher(state)

    mock_search.assert_awaited_once()
    assert result["web_docs"]
    assert any("vbpl.vn" in source for source in result["sources"])


@pytest.mark.asyncio
async def test_web_searcher_can_be_disabled_via_env(monkeypatch):
    from src.agent.retrieval import web_searcher

    state = _make_state("nghi dinh moi ve toc do toi da")
    state["intent"] = "speed"
    monkeypatch.setenv("ENABLE_WEB_FALLBACK", "false")

    with patch("src.agent.retrieval.search_official_web", new=AsyncMock(return_value=[])) as mock_search:
        result = await web_searcher(state)

    mock_search.assert_not_awaited()
    assert result["web_docs"] == []
