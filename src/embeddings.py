"""Embedding provider factory for ingestion and retrieval."""
import os
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


@lru_cache(maxsize=1)
def get_embedding_function():
    """Return embedding function based on EMBEDDING_PROVIDER."""
    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()

    if provider == "google":
        google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if google_api_key and not google_api_key.startswith("REPLACE_WITH_"):
            model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
            return GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=google_api_key,
            )
        print("[EMBEDDINGS] GOOGLE_API_KEY missing/placeholder, falling back to local model.")

    local_model = os.getenv(
        "LOCAL_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    return HuggingFaceEmbeddings(
        model_name=local_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
