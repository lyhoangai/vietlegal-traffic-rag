"""Regression tests for public portfolio hygiene."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_utf8(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _git_ls_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.splitlines())


def test_eval_dataset_uses_active_2025_sources_only():
    text = _read_utf8("src/eval/eval_dataset.json")
    ingest_test = _read_utf8("tests/test_ingest.py")

    assert "Ngh" in text
    assert "168/2024/N" in text
    assert "38/2024/TT-BGTVT" in text
    assert "100/2019" not in text
    assert "31/2019" not in text
    assert "2008" not in text
    assert "nghi_dinh_100_2019.pdf" not in ingest_test


def test_public_docs_do_not_pitch_legacy_regulations():
    demo = _read_utf8("docs/demo-script.md")
    readme = _read_utf8("README.md")
    architecture = _read_utf8("docs/architecture.md")

    for text in (demo, readme, architecture):
        assert "100/2019" not in text
        assert "31/2019" not in text
        assert "2008" not in text


def test_public_docs_are_utf8_clean():
    docs = [
        _read_utf8("README.md"),
        _read_utf8("docs/demo-script.md"),
        _read_utf8("docs/architecture.md"),
    ]

    mojibake_markers = ("Ã", "Ä", "Æ", "â€", "ðŸ")
    for text in docs:
        assert not any(marker in text for marker in mojibake_markers)


def test_public_branding_uses_traffic_rag_instead_of_ai_agent():
    readme = _read_utf8("README.md")
    demo = _read_utf8("docs/demo-script.md")
    architecture = _read_utf8("docs/architecture.md")
    api_main = _read_utf8("src/api/main.py")
    html = _read_utf8("src/api/static/index.html")

    for text in (readme, demo, architecture, api_main, html):
        assert "VietLegal Traffic RAG" in text
        assert "VietLegal AI Agent" not in text


def test_static_ui_keeps_public_branding_and_no_ascii_fallback_title():
    html = _read_utf8("src/api/static/index.html")

    assert "VietLegal Traffic RAG" in html
    assert "VietLegal AI</strong>" not in html
    assert "Tu van Luat Giao thong" not in html


def test_readme_has_compact_public_sections_and_no_internal_links():
    readme = _read_utf8("README.md")

    for heading in (
        "## Overview",
        "## Evidence",
        "## Architecture",
        "## Why This Is Trustworthy",
        "## Quick Start",
        "## API / Endpoints",
        "## Tests",
    ):
        assert heading in readme

    assert "docs/assets/chat-ui.png" in readme
    assert "docs/assets/history-sidebar.png" in readme
    assert "```mermaid" in readme
    assert "docs/assets/ci-badge.svg" in readme
    assert "docs/benchmarks/latest_summary.md" in readme
    assert "docs/benchmarks/latest_results.json" in readme
    assert "datasets/vietlegal-traffic-eval-v1/README.md" in readme
    assert "https://huggingface.co/datasets/lyhoang0104ls/vietlegal-traffic-eval-v1" in readme
    assert "docs/cv-bullets.md" not in readme
    assert "docs/github-checklist.md" not in readme
    assert "AGENTS.md" not in readme
    assert "CLAUDE.md" not in readme


def test_readme_uses_new_public_repo_slug():
    readme = _read_utf8("README.md")

    assert "lyhoangai/vietlegal-traffic-rag" in readme
    assert "lyhoangai/vietlegal-ai-agent" not in readme


def test_architecture_doc_uses_repo_relative_links_only():
    architecture = _read_utf8("docs/architecture.md")

    assert "D:\\" not in architecture
    assert "../src/agent/graph.py" in architecture
    assert "../src/api/main.py" in architecture
    assert "../src/memory/store.py" in architecture
    assert "../data/manifest.json" in architecture
    assert "../datasets/vietlegal-traffic-eval-v1/README.md" in architecture


def test_internal_only_files_are_not_tracked_in_public_repo():
    tracked = _git_ls_files()

    for path in (
        ".agent/AGENTS.md",
        "AGENTS.md",
        "CLAUDE.md",
        "docs/cv-bullets.md",
        "docs/github-checklist.md",
        "docs/plans/2026-03-13-vietlegal-agent-design.md",
        "docs/plans/2026-03-13-vietlegal-agent-plan.md",
        "docs/plans/task.md",
    ):
        assert path not in tracked


def test_gitignore_blocks_internal_and_local_only_artifacts():
    gitignore = _read_utf8(".gitignore")

    for path in (
        ".agent/",
        "AGENTS.md",
        "CLAUDE.md",
        ".claude/",
        "docs/plans/",
        "docs/cv-bullets.md",
        "docs/github-checklist.md",
        "docs/private/",
        "docs/benchmarks/smoke_results.json",
        "docs/benchmarks/smoke_summary.md",
        "uvicorn_run.err",
    ):
        assert path in gitignore


def test_repo_has_required_public_assets_and_dataset_package():
    required_paths = [
        "docs/assets/chat-ui.png",
        "docs/assets/history-sidebar.png",
        "docs/assets/ci-badge.svg",
        "datasets/vietlegal-traffic-eval-v1/README.md",
        "datasets/vietlegal-traffic-eval-v1/data.jsonl",
        "docs/benchmarks/latest_summary.md",
        "docs/benchmarks/latest_results.json",
        "docs/demo-script.md",
        "docs/architecture.md",
    ]

    for relative_path in required_paths:
        assert (ROOT / relative_path).exists(), relative_path


def test_ci_runs_pytest_and_benchmark_smoke():
    workflow = _read_utf8(".github/workflows/ci.yml")

    assert "pytest tests -q" in workflow
    assert "python -m src.eval.run_benchmark --smoke" in workflow
