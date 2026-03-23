"""Official-source web search helpers for legal freshness fallback."""

from __future__ import annotations

import json
import os
import re
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "manifest.json"
OFFICIAL_DOMAINS = (
    "vbpl.vn",
    "congbao.chinhphu.vn",
    "vanban.chinhphu.vn",
)
_SERPER_SEARCH_URL = "https://google.serper.dev/search"
_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_BING_RSS_URL = "https://www.bing.com/search"
_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "12"))
_CANONICAL_TITLES = {
    "luat_duong_bo_2024.pdf": "Luật Đường bộ 2024 (35/2024/QH15)",
    "luat_trat_tu_an_toan_giao_thong_duong_bo_2024.pdf": "Luật Trật tự, an toàn giao thông đường bộ 2024 (36/2024/QH15)",
    "nghi_dinh_168_2024.pdf": "Nghị định 168/2024/NĐ-CP",
    "thong_tu_38_2024_bgtvt.pdf": "Thông tư 38/2024/TT-BGTVT",
}


def _is_official_url(url: str) -> bool:
    if not url:
        return False
    hostname = (urlparse(url).hostname or "").lower()
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in OFFICIAL_DOMAINS)


def _domain_label(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname or "web"


def _clean_snippet(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    return cleaned[:500]


def _ascii_search_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = normalized.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", normalized).strip()


def _quoted_term(text: str) -> str:
    cleaned = _ascii_search_text(text)
    if not cleaned:
        return ""
    return f"\"{cleaned}\"" if " " in cleaned else cleaned


@lru_cache(maxsize=1)
def _load_active_manifest_entries() -> list[dict[str, Any]]:
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

    entries = []
    for entry in payload:
        if entry.get("status") != "active":
            continue
        source_url = str(entry.get("source_url") or "").strip()
        if not _is_official_url(source_url):
            continue
        entries.append(entry)
    return entries


def _local_fallback_summary(filename: str, *, query: str, intent: str, entities: dict | None = None) -> str:
    normalized_query = _ascii_search_text(query).lower()
    entities = entities or {}
    violation = _ascii_search_text(str(entities.get("violation_type") or "")).lower()
    combined = f"{normalized_query} {violation}".strip()

    if filename == "nghi_dinh_168_2024.pdf":
        if "den do" in combined or "den tin hieu" in combined:
            return (
                "Nguồn chính thống để đối chiếu mức xử phạt và trừ điểm giấy phép lái xe "
                "đối với hành vi không chấp hành hiệu lệnh của đèn tín hiệu giao thông."
            )
        if "mu bao hiem" in combined:
            return (
                "Nguồn chính thống để đối chiếu mức xử phạt lỗi không đội mũ bảo hiểm "
                "đối với người điều khiển xe mô tô, xe máy."
            )
        if "nong do con" in combined or "ruou bia" in combined:
            return (
                "Nguồn chính thống để đối chiếu các khung xử phạt nồng độ cồn "
                "đối với ô tô, xe tải hoặc xe máy."
            )
        return "Nguồn chính thống của nghị định xử phạt vi phạm hành chính về trật tự, an toàn giao thông đường bộ."

    if filename == "thong_tu_38_2024_bgtvt.pdf":
        if "khoang cach" in combined:
            return "Nguồn chính thống để đối chiếu quy định về khoảng cách an toàn giữa các xe."
        return "Nguồn chính thống để đối chiếu quy định về tốc độ khai thác tối đa và tối thiểu trên đường bộ."

    if filename == "luat_duong_bo_2024.pdf":
        return "Nguồn chính thống của Luật Đường bộ 2024, dùng để đối chiếu khung pháp lý nền về đường bộ."

    if filename == "luat_trat_tu_an_toan_giao_thong_duong_bo_2024.pdf":
        return "Nguồn chính thống của Luật Trật tự, an toàn giao thông đường bộ 2024, dùng để đối chiếu quy tắc giao thông."

    if intent == "penalty":
        return "Nguồn chính thống để đối chiếu quy định xử phạt giao thông đường bộ."
    if intent == "speed":
        return "Nguồn chính thống để đối chiếu quy định về tốc độ và khoảng cách an toàn."
    return "Nguồn chính thống để đối chiếu quy định giao thông đường bộ."


def _local_official_fallback(
    query: str,
    *,
    intent: str = "general",
    entities: dict | None = None,
    max_results: int = 3,
) -> list[dict[str, str]]:
    entries_by_file = {
        str(entry.get("filename") or ""): entry for entry in _load_active_manifest_entries()
    }

    if intent == "penalty":
        wanted_files = ["nghi_dinh_168_2024.pdf"]
    elif intent == "speed":
        wanted_files = [
            "thong_tu_38_2024_bgtvt.pdf",
            "luat_duong_bo_2024.pdf",
            "luat_trat_tu_an_toan_giao_thong_duong_bo_2024.pdf",
        ]
    else:
        wanted_files = [
            "luat_trat_tu_an_toan_giao_thong_duong_bo_2024.pdf",
            "luat_duong_bo_2024.pdf",
        ]

    results: list[dict[str, str]] = []
    for filename in wanted_files:
        entry = entries_by_file.get(filename)
        if not entry:
            continue
        url = str(entry.get("source_url") or "").strip()
        if not url:
            continue
        title = _CANONICAL_TITLES.get(filename) or str(entry.get("title") or filename)
        results.append(
            {
                "title": title,
                "url": url,
                "content": _local_fallback_summary(
                    filename,
                    query=query,
                    intent=intent,
                    entities=entities,
                ),
                "source": f"Web · {_domain_label(url)} · {title}",
            }
        )
        if len(results) >= max_results:
            break
    return results


def build_official_search_query(query: str, intent: str = "general", entities: dict | None = None) -> str:
    entities = entities or {}
    query_text = _ascii_search_text(query.strip())
    vehicle = _quoted_term(str(entities.get("vehicle_type") or "").strip())
    violation = _quoted_term(str(entities.get("violation_type") or "").strip())

    if intent == "penalty":
        parts = [
            "\"168/2024/ND-CP\"",
            violation,
            vehicle,
            "xu phat giao thong",
            query_text if not violation else "",
        ]
    elif intent == "speed":
        parts = [
            "\"38/2024/TT-BGTVT\"",
            "\"35/2024/QH15\"",
            "\"36/2024/QH15\"",
            violation or "\"toc do\"",
            "\"khoang cach an toan\"" if "khoang cach" in query_text else "",
            query_text if not violation else "",
        ]
    else:
        parts = [
            "\"35/2024/QH15\"",
            "\"36/2024/QH15\"",
            violation,
            vehicle,
            "quy tac giao thong",
            query_text if not violation else "",
        ]

    site_filter = " OR ".join(f"site:{domain}" for domain in OFFICIAL_DOMAINS)
    deduped_parts = []
    seen = set()
    for item in parts:
        normalized = item.lower()
        if not item or normalized in seen:
            continue
        seen.add(normalized)
        deduped_parts.append(item)
    return f"({site_filter}) " + " ".join(deduped_parts)


async def _search_with_tavily(
    query: str,
    *,
    intent: str = "general",
    entities: dict | None = None,
    max_results: int = 3,
) -> list[dict[str, str]]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []

    payload = {
        "api_key": api_key,
        "query": build_official_search_query(query, intent=intent, entities=entities),
        "search_depth": "advanced",
        "max_results": max_results,
        "include_domains": list(OFFICIAL_DOMAINS),
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
    }

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(_TAVILY_SEARCH_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    results: list[dict[str, str]] = []
    for item in data.get("results", []):
        url = (item.get("url") or "").strip()
        if not _is_official_url(url):
            continue
        title = (item.get("title") or "").strip() or url
        content = _clean_snippet(item.get("content") or item.get("snippet") or "")
        if not content:
            continue
        results.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "source": f"Web · {_domain_label(url)} · {title}",
            }
        )
    return results[:max_results]


async def _search_with_serper(
    query: str,
    *,
    intent: str = "general",
    entities: dict | None = None,
    max_results: int = 3,
) -> list[dict[str, str]]:
    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        return []

    payload = {
        "q": build_official_search_query(query, intent=intent, entities=entities),
        "gl": "vn",
        "hl": "vi",
        "num": max(3, min(max_results * 2, 10)),
    }
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(_SERPER_SEARCH_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    results: list[dict[str, str]] = []
    for item in data.get("organic", []):
        url = (item.get("link") or "").strip()
        if not _is_official_url(url):
            continue
        title = (item.get("title") or "").strip() or url
        content = _clean_snippet(item.get("snippet") or "")
        if not content:
            continue
        results.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "source": f"Web · {_domain_label(url)} · {title}",
            }
        )
        if len(results) >= max_results:
            break
    return results


async def _search_with_bing_rss(
    query: str,
    *,
    intent: str = "general",
    entities: dict | None = None,
    max_results: int = 3,
) -> list[dict[str, str]]:
    search_query = build_official_search_query(query, intent=intent, entities=entities)
    url = f"{_BING_RSS_URL}?format=rss&q={urllib.parse.quote(search_query)}"

    async with httpx.AsyncClient(timeout=_TIMEOUT, headers={"user-agent": "Mozilla/5.0"}) as client:
        response = await client.get(url)
        response.raise_for_status()

    root = ET.fromstring(response.text)
    results: list[dict[str, str]] = []
    for item in root.findall("./channel/item"):
        link = (item.findtext("link") or "").strip()
        if not _is_official_url(link):
            continue
        title = (item.findtext("title") or "").strip() or link
        description = _clean_snippet(item.findtext("description") or "")
        if not description:
            continue
        results.append(
            {
                "title": title,
                "url": link,
                "content": description,
                "source": f"Web · {_domain_label(link)} · {title}",
            }
        )
        if len(results) >= max_results:
            break
    return results


async def search_official_web(
    query: str,
    *,
    intent: str = "general",
    entities: dict | None = None,
    max_results: int = 3,
) -> list[dict[str, str]]:
    """Search official legal sources, preferring Serper, then Tavily, then Bing RSS."""
    try:
        results = await _search_with_serper(
            query,
            intent=intent,
            entities=entities,
            max_results=max_results,
        )
        if results:
            return results
    except Exception:
        pass

    try:
        results = await _search_with_tavily(
            query,
            intent=intent,
            entities=entities,
            max_results=max_results,
        )
        if results:
            return results
    except Exception:
        pass

    try:
        results = await _search_with_bing_rss(
            query,
            intent=intent,
            entities=entities,
            max_results=max_results,
        )
        if results:
            return results
    except Exception:
        pass

    return _local_official_fallback(
        query,
        intent=intent,
        entities=entities,
        max_results=max_results,
    )


def format_web_docs(results: list[dict[str, Any]]) -> list[str]:
    docs = []
    for item in results:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = _clean_snippet(str(item.get("content") or ""))
        if not content:
            continue
        docs.append(
            "\n".join(
                [
                    "[Nguồn web chính thống]",
                    f"Tiêu đề: {title or url}",
                    f"URL: {url}",
                    f"Tóm tắt: {content}",
                ]
            )
        )
    return docs


def format_web_sources(results: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("source") or "").strip() for item in results if item.get("source")]
