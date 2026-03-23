"""Ingestion pipeline tests."""

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document


def test_load_and_split_returns_chunks():
    from src.ingest.loader import load_and_split

    pdf_path = "data/nghi_dinh_168_2024.pdf"
    if not Path(pdf_path).exists():
        pytest.skip("PDF not found in data/")
    chunks = load_and_split(pdf_path, collection="traffic_penalties", title="Nghị định 168/2024")
    assert len(chunks) > 10
    assert all("source_file" in c.metadata for c in chunks)
    assert all("collection" in c.metadata for c in chunks)
    assert all(len(c.page_content) <= 1000 for c in chunks)
    assert all(c.metadata["collection"] == "traffic_penalties" for c in chunks)


def test_noise_filter_removes_signature_pages():
    from src.ingest.loader import _is_noise_page

    noisy = "Nơi nhận:\n- Thủ tướng Chính phủ\n- Bộ trưởng"
    clean = "Điều 5. Xử phạt hành vi vi phạm quy tắc giao thông đường bộ của người điều khiển xe mô tô."
    assert _is_noise_page(noisy) is True
    assert _is_noise_page(clean) is False


def test_active_manifest_entries_raise_for_missing_active_files(tmp_path):
    from src.ingest.loader import active_manifest_entries

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "filename": "missing.pdf",
                    "title": "Missing active doc",
                    "collection": "traffic_law",
                    "source_url": "https://example.com/missing",
                    "effective_date": "2025-01-01",
                    "status": "active",
                    "extractable_text": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        active_manifest_entries(data_dir=str(tmp_path), manifest_path=manifest_path, strict=True)


def test_load_all_pdfs_uses_manifest_and_skips_legacy(monkeypatch, tmp_path):
    from src.ingest.loader import load_all_pdfs

    active_file = tmp_path / "active.pdf"
    legacy_file = tmp_path / "legacy.pdf"
    active_file.write_bytes(b"%PDF-1.4")
    legacy_file.write_bytes(b"%PDF-1.4")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "filename": active_file.name,
                    "title": "Active doc",
                    "collection": "traffic_speed",
                    "source_url": "https://example.com/active",
                    "effective_date": "2025-01-01",
                    "status": "active",
                    "extractable_text": True,
                },
                {
                    "filename": legacy_file.name,
                    "title": "Legacy doc",
                    "collection": "traffic_law",
                    "source_url": "https://example.com/legacy",
                    "effective_date": "2019-01-01",
                    "status": "legacy",
                    "extractable_text": True,
                },
            ]
        ),
        encoding="utf-8",
    )

    def fake_load_and_split(pdf_path: str, collection: str | None = None, title: str | None = None):
        return [
            Document(
                page_content=f"content from {Path(pdf_path).name}",
                metadata={"source_file": Path(pdf_path).stem, "collection": collection, "title": title},
            )
        ]

    monkeypatch.setattr("src.ingest.loader.load_and_split", fake_load_and_split)

    by_collection = load_all_pdfs(data_dir=str(tmp_path), manifest_path=manifest_path, strict=True)

    assert list(by_collection.keys()) == ["traffic_speed"]
    assert by_collection["traffic_speed"][0].metadata["title"] == "Active doc"
