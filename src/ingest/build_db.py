"""Build multi-collection ChromaDB from the active corpus manifest."""

from __future__ import annotations

import os
import shutil
import sys
import time

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

from src.embeddings import get_embedding_function
from src.ingest.loader import MANIFEST_PATH, load_all_pdfs

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
BATCH_SIZE = 50


def build_vector_db(
    data_dir: str = "data",
    manifest_path: str = str(MANIFEST_PATH),
    statuses: tuple[str, ...] = ("active",),
) -> dict:
    """Build one ChromaDB collection per active document type."""
    chunks_by_col = load_all_pdfs(
        data_dir=data_dir,
        manifest_path=manifest_path,
        statuses=statuses,
        strict=True,
    )
    if not chunks_by_col:
        raise ValueError("No active extractable corpus files found in data/manifest.json")

    embeddings = get_embedding_function()

    dbs = {}
    for collection, chunks in chunks_by_col.items():
        col_path = os.path.join(CHROMA_PATH, collection)
        if os.path.exists(col_path):
            shutil.rmtree(col_path)

        print(f"\nBuilding collection '{collection}' ({len(chunks)} chunks)...")
        db = None
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            if db is None:
                db = Chroma.from_documents(
                    batch,
                    embeddings,
                    persist_directory=col_path,
                    collection_name=collection,
                )
            else:
                db.add_documents(batch)
            print(f"  Batch {i // BATCH_SIZE + 1}/{(len(chunks) - 1) // BATCH_SIZE + 1} done")
            time.sleep(1)

        dbs[collection] = db
        print(f"  -> '{collection}' built at {col_path}")

    return dbs


if __name__ == "__main__":
    build_vector_db()
