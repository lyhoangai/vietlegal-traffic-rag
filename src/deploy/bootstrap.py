"""Container bootstrap helpers for local Docker and Render deploys."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def build_vector_db():
    """Build the Chroma store lazily so imports stay lightweight."""
    from src.ingest.build_db import build_vector_db as _build_vector_db

    return _build_vector_db()


def _chroma_path() -> Path:
    return Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))


def _vector_db_ready(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def ensure_vector_db() -> bool:
    """Build the vector store on first boot and skip later restarts."""
    chroma_path = _chroma_path()
    if _vector_db_ready(chroma_path):
        print(f"[BOOTSTRAP] Reusing existing vector store at {chroma_path}")
        return False

    print(f"[BOOTSTRAP] No vector store found at {chroma_path}; building now...")
    chroma_path.mkdir(parents=True, exist_ok=True)
    build_vector_db()
    return True


def build_uvicorn_command() -> list[str]:
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8000")
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        host,
        "--port",
        port,
    ]


def main() -> None:
    ensure_vector_db()
    subprocess.run(build_uvicorn_command(), check=True)


if __name__ == "__main__":
    main()
