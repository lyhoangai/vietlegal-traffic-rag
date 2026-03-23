"""Unit tests for official web-search helpers."""

from unittest.mock import AsyncMock, patch

import pytest

from src.web_search import build_official_search_query, format_web_docs


def test_build_official_search_query_includes_official_domains_and_intent_hint():
    query = build_official_search_query(
        "nghi dinh moi ve vuot den do",
        intent="penalty",
        entities={"vehicle_type": "o to", "violation_type": "vuot den do"},
    )

    assert "site:vbpl.vn" in query
    assert "site:congbao.chinhphu.vn" in query
    assert "168/2024/ND-CP" in query
    assert "vuot den do" in query


def test_format_web_docs_generates_prompt_ready_blocks():
    docs = format_web_docs(
        [
            {
                "title": "Nghi dinh moi",
                "url": "https://congbao.chinhphu.vn/van-ban/nghi-dinh-moi.htm",
                "content": "Noi dung cap nhat.",
            }
        ]
    )

    assert len(docs) == 1
    assert "[Nguồn web chính thống]" in docs[0]
    assert "URL: https://congbao.chinhphu.vn/van-ban/nghi-dinh-moi.htm" in docs[0]
    assert "Tóm tắt: Noi dung cap nhat." in docs[0]


@pytest.mark.asyncio
async def test_search_official_web_uses_manifest_fallback_for_penalty_when_remote_search_fails():
    from src.web_search import search_official_web

    with patch("src.web_search._search_with_serper", new=AsyncMock(return_value=[])), patch(
        "src.web_search._search_with_tavily", new=AsyncMock(return_value=[])
    ), patch("src.web_search._search_with_bing_rss", new=AsyncMock(return_value=[])):
        results = await search_official_web(
            "doi chieu nguon chinh thong muc phat o to vuot den do",
            intent="penalty",
            entities={"vehicle_type": "o to", "violation_type": "vuot den do"},
            max_results=3,
        )

    assert results
    assert any("168/2024" in item["title"] for item in results)
    assert any("congbao.chinhphu.vn" in item["url"] for item in results)
    assert any("xử phạt" in item["content"].lower() for item in results)


@pytest.mark.asyncio
async def test_search_official_web_uses_manifest_fallback_for_speed_when_remote_search_fails():
    from src.web_search import search_official_web

    with patch("src.web_search._search_with_serper", new=AsyncMock(return_value=[])), patch(
        "src.web_search._search_with_tavily", new=AsyncMock(return_value=[])
    ), patch("src.web_search._search_with_bing_rss", new=AsyncMock(return_value=[])):
        results = await search_official_web(
            "doi chieu nguon chinh thong toc do toi da tren duong cao toc",
            intent="speed",
            entities={"violation_type": "toc do"},
            max_results=3,
        )

    assert results
    assert any("38/2024/TT-BGTVT" in item["title"] for item in results)
    assert any("congbao.chinhphu.vn" in item["url"] for item in results)
    assert any("tốc độ" in item["content"].lower() for item in results)
