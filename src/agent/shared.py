"""Shared agent helpers and constants for the VietLegal pipeline."""

from __future__ import annotations

import os
import re

from langchain_community.vectorstores import Chroma

from src.agent.text_utils import (
    _is_out_of_scope_query,
    _normalize_text,
    _query_requests_web_confirmation,
    _query_requests_web_freshness,
)
from src.embeddings import get_embedding_function

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

_dbs: dict = {}
_all_docs_cache: dict = {}
_LEGAL_CITATION_RE = re.compile(
    r"\b(\u0110i\u1ec1u|Kho\u1ea3n|Ngh\u1ecb \u0111\u1ecbnh|Th\u00f4ng t\u01b0|Lu\u1eadt)\b",
    re.IGNORECASE,
)
_VAGUE_PHRASES = [
    "kh\u00f4ng c\u00f3 th\u00f4ng tin c\u1ee5 th\u1ec3",
    "kh\u00f4ng \u0111\u01b0\u1ee3c ch\u1ec9 \u0111\u1ecbnh r\u00f5",
    "th\u00f4ng th\u01b0\u1eddng",
    "c\u00f3 th\u1ec3 tham kh\u1ea3o",
    "khuy\u1ebfn ngh\u1ecb b\u1ea1n tham kh\u1ea3o",
]

def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _answer_with_sources(answer: str, sources: list, limit: int = 3) -> str:
    return answer


def _get_db(collection: str) -> Chroma:
    if collection not in _dbs:
        col_path = os.path.join(CHROMA_PATH, collection)
        embeddings = get_embedding_function()
        _dbs[collection] = Chroma(
            persist_directory=col_path,
            embedding_function=embeddings,
            collection_name=collection,
        )
    return _dbs[collection]


def _dedupe_documents(docs: list) -> list:
    unique = []
    seen = set()
    for doc in docs:
        key = (
            doc.metadata.get("source_file", ""),
            doc.metadata.get("page", -1),
            doc.page_content[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _dedupe_sources(sources: list) -> list:
    unique = []
    seen = set()
    for source in sources:
        if source in seen:
            continue
        seen.add(source)
        unique.append(source)
    return unique


def _needs_conservative_fallback(answer: str) -> bool:
    lower = (answer or "").lower()
    if not _LEGAL_CITATION_RE.search(answer or ""):
        return True
    return any(phrase in lower for phrase in _VAGUE_PHRASES)


def _build_conservative_answer(intent: str, sources: list) -> str:
    if intent == "penalty":
        return (
            "Hi\u1ec7n ch\u01b0a \u0111\u1ee7 c\u0103n c\u1ee9 \u0111\u1ec3 k\u1ebft lu\u1eadn m\u1ee9c ph\u1ea1t ch\u00ednh x\u00e1c t\u1eeb c\u00e1c \u0111o\u1ea1n tr\u00edch \u0111\u00e3 truy xu\u1ea5t. "
            "Vui l\u00f2ng \u0111\u1ed1i chi\u1ebfu tr\u1ef1c ti\u1ebfp v\u0103n b\u1ea3n g\u1ed1c v\u00e0 \u0111i\u1ec1u kho\u1ea3n li\u00ean quan tr\u01b0\u1edbc khi \u00e1p d\u1ee5ng."
        )
    return (
        "Hi\u1ec7n ch\u01b0a \u0111\u1ee7 c\u0103n c\u1ee9 ph\u00e1p l\u00fd r\u00f5 r\u00e0ng trong ng\u1eef c\u1ea3nh truy xu\u1ea5t \u0111\u1ec3 tr\u1ea3 l\u1eddi d\u1ee9t kho\u00e1t. "
        "Vui l\u00f2ng ki\u1ec3m tra l\u1ea1i v\u0103n b\u1ea3n g\u1ed1c."
    )


def _build_scope_limited_answer() -> str:
    return (
        "Demo hi\u1ec7n t\u1ea1i ch\u1ec9 t\u1eadp trung v\u00e0o 3 nh\u00f3m c\u00e2u h\u1ecfi: m\u1ee9c ph\u1ea1t vi ph\u1ea1m giao th\u00f4ng, "
        "quy t\u1eafc tham gia giao th\u00f4ng, v\u00e0 quy \u0111\u1ecbnh v\u1ec1 t\u1ed1c \u0111\u1ed9/kho\u1ea3ng c\u00e1ch an to\u00e0n. "
        "C\u00e2u h\u1ecfi c\u1ee7a b\u1ea1n \u0111ang n\u1eb1m ngo\u00e0i ph\u1ea1m vi demo n\u00e0y n\u00ean m\u00ecnh kh\u00f4ng tr\u1ea3 l\u1eddi d\u01b0\u1edbi d\u1ea1ng k\u1ebft lu\u1eadn ph\u00e1p l\u00fd."
    )


