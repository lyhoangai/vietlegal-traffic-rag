"""Tests for the public benchmark dataset package."""

from __future__ import annotations

import json

from src.eval.benchmark_dataset import (
    ALLOWED_CATEGORIES,
    DATASET_CARD_PATH,
    DATASET_PATH,
    LEGACY_MARKERS,
    REQUIRED_FIELDS,
    load_benchmark_dataset,
)


def test_benchmark_dataset_has_publish_ready_schema():
    entries = load_benchmark_dataset()

    assert 30 <= len(entries) <= 300

    seen_ids = set()
    seen_categories = set()

    for entry in entries:
        assert REQUIRED_FIELDS.issubset(entry)
        assert entry["id"] not in seen_ids
        assert entry["category"] in ALLOWED_CATEGORIES
        assert entry["expected_behavior"] in {"answer", "refuse"}

        seen_ids.add(entry["id"])
        seen_categories.add(entry["category"])

        if entry["category"] == "follow_up":
            assert entry.get("messages")

        if entry["expected_behavior"] == "answer":
            assert entry.get("required_phrases")

        payload = json.dumps(entry, ensure_ascii=False)
        for legacy_marker in LEGACY_MARKERS:
            assert legacy_marker not in payload

    assert seen_categories == set(ALLOWED_CATEGORIES)


def test_benchmark_dataset_includes_official_source_verification_case():
    entries = load_benchmark_dataset()

    matching = [
        entry
        for entry in entries
        if entry["expected_behavior"] == "answer"
        and "nguon chinh thong" in entry["question"].lower()
        and any(
            "da doi chieu nguon web chinh thong" in phrase.lower()
            for phrase in entry.get("required_phrases", [])
        )
    ]

    assert matching, "benchmark dataset should include at least one explicit official-source verification case"


def test_dataset_card_documents_scope_and_publish_steps():
    assert DATASET_PATH.exists()
    assert DATASET_CARD_PATH.exists()

    card = DATASET_CARD_PATH.read_text(encoding="utf-8")

    assert "Hugging Face" in card
    assert "huggingface-cli" in card
    assert "scope" in card.lower()
    assert DATASET_PATH.name in card
