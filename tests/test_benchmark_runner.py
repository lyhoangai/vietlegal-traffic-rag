"""Tests for benchmark runner outputs and smoke behavior."""

from __future__ import annotations

import json

import pytest


def _write_jsonl(path, rows):
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


@pytest.mark.asyncio
async def test_run_benchmark_writes_json_and_markdown(tmp_path):
    from src.eval.run_benchmark import run_benchmark

    dataset_path = tmp_path / "benchmark.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "id": "penalty-1",
                "category": "penalty",
                "question": "O to vuot den do phat bao nhieu?",
                "reference_answer": "Phat tu 18.000.000 dong den 20.000.000 dong.",
                "expected_behavior": "answer",
                "primary_source": "Nghi dinh 168/2024/ND-CP",
                "required_phrases": [
                    "18.000.000 dong den 20.000.000 dong",
                    "Nghi dinh 168/2024/ND-CP",
                ],
            },
            {
                "id": "scope-1",
                "category": "scope_refusal",
                "question": "Doi bang lai xe can gi?",
                "reference_answer": "Ngoai pham vi demo.",
                "expected_behavior": "refuse",
                "primary_source": "README supported scope",
                "required_phrases": ["ngoai pham vi demo"],
            },
        ],
    )

    async def fake_invoke_case(entry, *, mode_name, llm_provider):
        if entry["expected_behavior"] == "refuse":
            return {
                "answer": "Cau hoi nay dang nam ngoai pham vi demo.",
                "confidence": 0.2,
                "sources": [],
                "web_used": False,
            }

        answer = "Phat tu 18.000.000 dong den 20.000.000 dong theo Nghi dinh 168/2024/ND-CP."
        if mode_name == "no_reranker":
            answer = "Phat tu 18.000.000 dong den 20.000.000 dong."
        return {
            "answer": answer,
            "confidence": 0.9,
            "sources": ["nghi_dinh_168_2024 trang 16"],
            "web_used": mode_name == "no_web_fallback",
        }

    report = await run_benchmark(
        dataset_path=dataset_path,
        output_dir=tmp_path,
        invoke_case=fake_invoke_case,
    )

    results_path = tmp_path / "latest_results.json"
    summary_path = tmp_path / "latest_summary.md"

    assert results_path.exists()
    assert summary_path.exists()
    assert report["artifacts"]["results_json"] == str(results_path)
    assert report["artifacts"]["summary_markdown"] == str(summary_path)

    raw = json.loads(results_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")

    assert set(raw["modes"]) == {"full", "no_reranker", "no_web_fallback"}
    assert raw["dataset_path"] == dataset_path.resolve().as_posix()
    assert "| Mode | Cases | Pass Rate |" in summary
    assert "no_reranker" in summary
    assert "scope_refusal" in summary
    assert "# VietLegal Traffic RAG Benchmark Summary" in summary
    assert f"`{dataset_path.resolve().as_posix()}`" in summary


@pytest.mark.asyncio
async def test_smoke_benchmark_writes_smoke_artifacts_without_overwriting_latest(tmp_path):
    from src.eval.run_benchmark import run_benchmark

    dataset_path = tmp_path / "benchmark.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "id": "penalty-1",
                "category": "penalty",
                "question": "O to vuot den do phat bao nhieu?",
                "reference_answer": "Phat tu 18.000.000 dong den 20.000.000 dong.",
                "expected_behavior": "answer",
                "primary_source": "Nghi dinh 168/2024/ND-CP",
                "required_phrases": [
                    "18.000.000 dong den 20.000.000 dong",
                    "Nghi dinh 168/2024/ND-CP",
                ],
            },
            {
                "id": "speed-1",
                "category": "speed",
                "question": "Cao toc duoc chay 120 km/h khong?",
                "reference_answer": "Toc do toi da la 120 km/h.",
                "expected_behavior": "answer",
                "primary_source": "Thong tu 38/2024/TT-BGTVT",
                "required_phrases": [
                    "120 km/h",
                    "Thong tu 38/2024/TT-BGTVT",
                ],
            },
            {
                "id": "follow-up-1",
                "category": "follow_up",
                "question": "Con xe may thi sao?",
                "reference_answer": "Xe may co muc phat rieng.",
                "expected_behavior": "answer",
                "primary_source": "Nghi dinh 168/2024/ND-CP",
                "required_phrases": [
                    "Nghi dinh 168/2024/ND-CP",
                ],
                "messages": [{"role": "user", "content": "O to vuot den do phat bao nhieu?"}],
            },
            {
                "id": "scope-1",
                "category": "scope_refusal",
                "question": "Doi bang lai xe can gi?",
                "reference_answer": "Ngoai pham vi demo.",
                "expected_behavior": "refuse",
                "primary_source": "README supported scope",
                "required_phrases": ["ngoai pham vi demo"],
            },
        ],
    )

    async def fake_invoke_case(entry, *, mode_name, llm_provider):
        if entry["expected_behavior"] == "refuse":
            return {
                "answer": "Cau hoi nay dang nam ngoai pham vi demo.",
                "confidence": 0.2,
                "sources": [],
                "web_used": False,
            }

        return {
            "answer": f"{entry['reference_answer']} Theo {entry['primary_source']}.",
            "confidence": 0.9,
            "sources": [entry["primary_source"]],
            "web_used": False,
        }

    await run_benchmark(
        dataset_path=dataset_path,
        output_dir=tmp_path,
        invoke_case=fake_invoke_case,
    )

    latest_summary = tmp_path / "latest_summary.md"
    latest_results = tmp_path / "latest_results.json"
    latest_summary.write_text("full benchmark artifact", encoding="utf-8")
    latest_results.write_text('{"kind":"full"}', encoding="utf-8")

    report = await run_benchmark(
        dataset_path=dataset_path,
        output_dir=tmp_path,
        smoke=True,
        invoke_case=fake_invoke_case,
    )

    smoke_summary = tmp_path / "smoke_summary.md"
    smoke_results = tmp_path / "smoke_results.json"

    assert smoke_summary.exists()
    assert smoke_results.exists()
    assert latest_summary.read_text(encoding="utf-8") == "full benchmark artifact"
    assert latest_results.read_text(encoding="utf-8") == '{"kind":"full"}'
    assert report["artifacts"]["summary_markdown"] == str(smoke_summary)
    assert report["artifacts"]["results_json"] == str(smoke_results)
    raw = json.loads(smoke_results.read_text(encoding="utf-8"))
    summary = smoke_summary.read_text(encoding="utf-8")
    assert raw["dataset_path"] == dataset_path.resolve().as_posix()
    assert f"`{dataset_path.resolve().as_posix()}`" in summary


@pytest.mark.asyncio
async def test_run_benchmark_continues_after_case_error_and_records_it(tmp_path):
    from src.eval.run_benchmark import run_benchmark

    dataset_path = tmp_path / "benchmark.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "id": "penalty-1",
                "category": "penalty",
                "question": "O to vuot den do phat bao nhieu?",
                "reference_answer": "Phat tu 18.000.000 dong den 20.000.000 dong.",
                "expected_behavior": "answer",
                "primary_source": "Nghi dinh 168/2024/ND-CP",
                "required_phrases": [
                    "18.000.000 dong den 20.000.000 dong",
                    "Nghi dinh 168/2024/ND-CP",
                ],
            },
            {
                "id": "speed-1",
                "category": "speed",
                "question": "Cao toc duoc chay 120 km/h khong?",
                "reference_answer": "Toc do toi da la 120 km/h.",
                "expected_behavior": "answer",
                "primary_source": "Thong tu 38/2024/TT-BGTVT",
                "required_phrases": [
                    "120 km/h",
                    "Thong tu 38/2024/TT-BGTVT",
                ],
            },
        ],
    )

    async def fake_invoke_case(entry, *, mode_name, llm_provider):
        if entry["id"] == "penalty-1":
            raise RuntimeError("gemini quota exceeded")
        return {
            "answer": "Toc do toi da la 120 km/h theo Thong tu 38/2024/TT-BGTVT.",
            "confidence": 0.9,
            "sources": ["Thong tu 38/2024/TT-BGTVT"],
            "web_used": False,
        }

    report = await run_benchmark(
        dataset_path=dataset_path,
        output_dir=tmp_path,
        modes=("full",),
        invoke_case=fake_invoke_case,
    )

    rows = report["modes"]["full"]
    assert len(rows) == 2

    failed = next(row for row in rows if row["id"] == "penalty-1")
    succeeded = next(row for row in rows if row["id"] == "speed-1")

    assert failed["passed"] is False
    assert failed["answer"] == ""
    assert failed["error"] == "RuntimeError: gemini quota exceeded"

    assert succeeded["passed"] is True
    assert succeeded["error"] is None

    assert report["summary"]["full"]["totals"]["errors"] == 1

    summary = (tmp_path / "latest_summary.md").read_text(encoding="utf-8")
    assert "| Mode | Cases | Pass Rate | Errors |" in summary
    assert "| penalty | 1 | 0.0% | 1 |" in summary


def test_score_case_matches_ascii_rubric_against_vietnamese_answer():
    from src.eval.run_benchmark import _score_case

    entry = {
        "expected_behavior": "answer",
        "primary_source": "Dieu 6 Nghi dinh 168/2024/ND-CP",
        "required_phrases": [
            "18.000.000 dong den 20.000.000 dong",
            "04 diem",
            "Dieu 6",
            "168/2024/ND-CP",
        ],
    }
    response = {
        "answer": (
            "Ô tô vượt đèn đỏ: phạt từ 18.000.000 đồng đến 20.000.000 đồng, "
            "kèm trừ 04 điểm giấy phép lái xe.\n"
            "Căn cứ: điểm b khoản 9 và điểm b khoản 16 Điều 6 "
            "Nghị định 168/2024/NĐ-CP."
        ),
        "sources": ["Web · congbao.chinhphu.vn · Nghị định 168/2024/NĐ-CP"],
    }

    scored = _score_case(entry, response)

    assert scored["matched_phrases"] == entry["required_phrases"]
    assert scored["phrase_match_rate"] == 1.0
    assert scored["source_hit"] is True
    assert scored["passed"] is True


def test_smoke_case_selection_keeps_one_case_per_category():
    from src.eval.run_benchmark import select_smoke_cases

    entries = [
        {
            "id": f"{category}-{index}",
            "category": category,
            "question": f"question {index}",
            "reference_answer": "answer",
            "expected_behavior": "answer" if category != "scope_refusal" else "refuse",
            "primary_source": "source",
        }
        for category in ("penalty", "speed", "follow_up", "scope_refusal")
        for index in range(2)
    ]

    smoke_cases = select_smoke_cases(entries)

    assert len(smoke_cases) == 4
    assert {entry["category"] for entry in smoke_cases} == {
        "penalty",
        "speed",
        "follow_up",
        "scope_refusal",
    }


def test_display_path_uses_repo_relative_path_for_portfolio_dataset():
    from src.eval.run_benchmark import DATASET_PATH, _display_path

    assert _display_path(DATASET_PATH) == "datasets/vietlegal-traffic-eval-v2/data.jsonl"
