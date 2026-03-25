from __future__ import annotations

import subprocess


def test_ensure_vector_db_builds_when_store_is_missing(tmp_path, monkeypatch):
    from src.deploy import bootstrap

    chroma_path = tmp_path / "chroma_db"
    built = []

    monkeypatch.setenv("CHROMA_DB_PATH", str(chroma_path))
    monkeypatch.setattr(bootstrap, "build_vector_db", lambda: built.append("built"))

    assert bootstrap.ensure_vector_db() is True
    assert built == ["built"]


def test_ensure_vector_db_skips_build_when_store_exists(tmp_path, monkeypatch):
    from src.deploy import bootstrap

    chroma_path = tmp_path / "chroma_db"
    (chroma_path / "traffic_law").mkdir(parents=True)
    (chroma_path / "traffic_law" / "marker.txt").write_text("ready", encoding="utf-8")
    built = []

    monkeypatch.setenv("CHROMA_DB_PATH", str(chroma_path))
    monkeypatch.setattr(bootstrap, "build_vector_db", lambda: built.append("built"))

    assert bootstrap.ensure_vector_db() is False
    assert built == []


def test_build_uvicorn_command_uses_env_port(monkeypatch):
    from src.deploy import bootstrap

    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("HOST", "0.0.0.0")

    command = bootstrap.build_uvicorn_command()

    assert command[:3] == [bootstrap.sys.executable, "-m", "uvicorn"]
    assert "src.api.main:app" in command
    assert command[-2:] == ["--port", "9000"]


def test_main_runs_bootstrap_before_starting_server(monkeypatch):
    from src.deploy import bootstrap

    events: list[str | tuple[str, list[str], bool]] = []

    monkeypatch.setattr(
        bootstrap,
        "ensure_vector_db",
        lambda: events.append("ensure") or False,
    )
    monkeypatch.setattr(
        bootstrap,
        "build_uvicorn_command",
        lambda: ["python", "-m", "uvicorn", "src.api.main:app"],
    )

    def fake_run(command: list[str], check: bool):
        events.append(("run", command, check))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    bootstrap.main()

    assert events == [
        "ensure",
        ("run", ["python", "-m", "uvicorn", "src.api.main:app"], True),
    ]
