"""Load and split corpus files into Chroma-ready chunks using a manifest."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SEPARATORS = ["\nĐiều ", "\nKhoản ", "\nĐiểm ", "\n\n", "\n"]
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "manifest.json"

NOISE_PHRASES = ["Nơi nhận:", "TM. CHÍNH PHỦ", "TM. BỘ GIAO THÔNG", "KT. BỘ TRƯỞNG"]


def _is_noise_page(text: str) -> bool:
    """Return True if a page is a signature/header-only page."""
    return any(phrase in text for phrase in NOISE_PHRASES) and len(text) < 800


def load_manifest(manifest_path: str | Path = MANIFEST_PATH) -> list[dict]:
    """Read the corpus manifest from disk."""
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, list):
        raise ValueError("Corpus manifest must be a list of document entries")
    return manifest


def active_manifest_entries(
    data_dir: str = "data",
    manifest_path: str | Path = MANIFEST_PATH,
    statuses: tuple[str, ...] = ("active",),
    strict: bool = True,
) -> list[dict]:
    """Return manifest entries that should be ingested for the requested statuses."""
    root = Path(data_dir)
    entries = []
    missing = []

    for entry in load_manifest(manifest_path):
        if entry.get("status") not in statuses:
            continue
        if not entry.get("extractable_text", False):
            continue
        resolved = root / entry["filename"]
        record = {**entry, "path": str(resolved)}
        entries.append(record)
        if not resolved.exists():
            missing.append(entry["filename"])

    if strict and missing:
        joined = ", ".join(sorted(missing))
        raise FileNotFoundError(
            "Missing active corpus files in data/: "
            f"{joined}. Add the 2025 corpus files listed in data/manifest.json before rebuilding Chroma."
        )
    return [entry for entry in entries if Path(entry["path"]).exists()]


def load_and_split(pdf_path: str, collection: str | None = None, title: str | None = None) -> list:
    """Load a PDF and split into chunks with metadata."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = splitter.split_documents(pages)

    source_name = Path(pdf_path).stem
    filtered = []
    for chunk in chunks:
        if _is_noise_page(chunk.page_content):
            continue
        chunk.metadata["source_file"] = source_name
        if collection:
            chunk.metadata["collection"] = collection
        if title:
            chunk.metadata["title"] = title
        filtered.append(chunk)
    return filtered


def load_all_pdfs(
    data_dir: str = "data",
    manifest_path: str | Path = MANIFEST_PATH,
    statuses: tuple[str, ...] = ("active",),
    strict: bool = True,
) -> dict:
    """Load manifest-selected PDFs and group chunks by collection name."""
    by_collection: dict = {}

    for entry in active_manifest_entries(
        data_dir=data_dir,
        manifest_path=manifest_path,
        statuses=statuses,
        strict=strict,
    ):
        pdf = Path(entry["path"])
        print(f"Loading {pdf.name}...")
        chunks = load_and_split(
            str(pdf),
            collection=entry["collection"],
            title=entry.get("title"),
        )
        print(f"  -> {len(chunks)} usable chunks")
        for chunk in chunks:
            col = chunk.metadata.get("collection", entry["collection"])
            by_collection.setdefault(col, []).append(chunk)

    for col, docs in by_collection.items():
        print(f"Collection '{col}': {len(docs)} chunks total")

    return by_collection
