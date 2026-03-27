"""Helpers for the publishable benchmark dataset package."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "datasets" / "vietlegal-traffic-eval-v2"
DATASET_PATH = DATASET_DIR / "data.jsonl"
DATASET_CARD_PATH = DATASET_DIR / "README.md"

ALLOWED_CATEGORIES = (
    "penalty",
    "speed",
    "follow_up",
    "scope_refusal",
)
REQUIRED_FIELDS = frozenset(
    {
        "id",
        "category",
        "question",
        "reference_answer",
        "expected_behavior",
        "primary_source",
    }
)
LEGACY_MARKERS = (
    "Nghi dinh 100/2019",
    "Thong tu 31/2019",
    "Luat Giao thong duong bo 2008",
    "nghi_dinh_100_2019",
    "thong_tu_31_2019",
    "luat_giao_thong_duong_bo_2008",
)


def _read_jsonl(path: Path) -> list[dict]:
    entries = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        entries.append(json.loads(line))
    return entries


def validate_benchmark_dataset(entries: list[dict], *, require_publish_ready: bool = False) -> None:
    errors = []
    seen_ids = set()
    seen_categories = set()

    if require_publish_ready and not 30 <= len(entries) <= 300:
        errors.append("publish-ready dataset must contain 30-300 rows")

    for index, entry in enumerate(entries, start=1):
        missing = REQUIRED_FIELDS.difference(entry)
        if missing:
            errors.append(f"row {index} missing fields: {sorted(missing)}")
            continue

        entry_id = str(entry["id"])
        if entry_id in seen_ids:
            errors.append(f"duplicate id: {entry_id}")
        seen_ids.add(entry_id)

        category = entry["category"]
        if category not in ALLOWED_CATEGORIES:
            errors.append(f"row {index} has invalid category: {category}")
        else:
            seen_categories.add(category)

        expected_behavior = entry["expected_behavior"]
        if expected_behavior not in {"answer", "refuse"}:
            errors.append(f"row {index} has invalid expected_behavior: {expected_behavior}")

        if category == "follow_up" and not entry.get("messages"):
            errors.append(f"row {index} must include messages for follow_up cases")

        if expected_behavior == "answer" and not entry.get("required_phrases"):
            errors.append(f"row {index} must include required_phrases for answer cases")

        payload = json.dumps(entry, ensure_ascii=False)
        for legacy_marker in LEGACY_MARKERS:
            if legacy_marker in payload:
                errors.append(f"row {index} contains legacy marker: {legacy_marker}")

    if require_publish_ready:
        missing_categories = set(ALLOWED_CATEGORIES).difference(seen_categories)
        if missing_categories:
            errors.append(f"dataset missing categories: {sorted(missing_categories)}")

    if errors:
        raise ValueError("; ".join(errors))


def load_benchmark_dataset(
    path: str | Path | None = None,
    *,
    require_publish_ready: bool | None = None,
) -> list[dict]:
    dataset_path = Path(path) if path else DATASET_PATH
    if require_publish_ready is None:
        require_publish_ready = dataset_path.resolve() == DATASET_PATH.resolve()

    entries = _read_jsonl(dataset_path)
    validate_benchmark_dataset(entries, require_publish_ready=require_publish_ready)
    return entries
