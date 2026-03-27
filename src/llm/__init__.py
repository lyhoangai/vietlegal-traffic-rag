"""src/llm/__init__.py — LLM factory with cascading fallbacks & SQLite cache."""
import os
import sqlite3
import hashlib
from src.llm.gemini import GeminiLLM
from src.llm.groq_client import GroqLLM
from src.llm.openrouter import OpenRouterLLM

CACHE_DB = ".llm_cache.db"


def _init_cache():
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, response TEXT)")

_init_cache()


def _get_cached(prompt: str, provider: str):
    h = hashlib.md5(f"{provider}:{prompt}".encode()).hexdigest()
    with sqlite3.connect(CACHE_DB) as conn:
        res = conn.execute("SELECT response FROM cache WHERE hash=?", (h,)).fetchone()
    return res[0] if res else None


def _set_cached(prompt: str, provider: str, response: str):
    h = hashlib.md5(f"{provider}:{prompt}".encode()).hexdigest()
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute("INSERT OR REPLACE INTO cache (hash, response) VALUES (?, ?)", (h, response))


def get_llm(provider: str = None):
    """Return an LLM by provider name."""
    provider = provider or os.getenv("LLM_PROVIDER", "gemini")
    if provider == "groq":
        return GroqLLM()
    if provider == "openrouter":
        return OpenRouterLLM()
    return GeminiLLM()


async def invoke_with_fallback(prompt: str, state: dict) -> str:
    """Call preferred provider first; then fallback to others, using SQLite Cache."""
    preferred = (state or {}).get("llm_provider") or os.getenv("LLM_PROVIDER", "gemini")
    
    order = ["groq", "gemini", "openrouter"]
    if preferred == "gemini":
        order = ["gemini", "groq", "openrouter"]
    elif preferred == "openrouter":
        order = ["openrouter", "gemini", "groq"]

    last_error = None

    for provider in order:
        cached = _get_cached(prompt, provider)
        if cached:
            state["llm_provider"] = provider
            print(f"[CACHE HIT] {provider} returned from cache.")
            return cached

        try:
            llm = get_llm(provider)
            result = await llm.ainvoke(prompt)
            _set_cached(prompt, provider, result)
            state["llm_provider"] = provider
            return result
        except Exception as err:
            print(f"[LLM FALLBACK] {provider} failed ({err})")
            last_error = err

    raise last_error if last_error else RuntimeError("No LLM provider available")


async def stream_with_fallback(prompt: str, state: dict):
    """Stream with preferred provider first; fallback to others."""
    preferred = (state or {}).get("llm_provider") or os.getenv("LLM_PROVIDER", "gemini")
    
    order = ["groq", "gemini", "openrouter"]
    if preferred == "gemini":
        order = ["gemini", "groq", "openrouter"]
    elif preferred == "openrouter":
        order = ["openrouter", "gemini", "groq"]

    last_error = None

    for provider in order:
        try:
            llm = get_llm(provider)
            state["llm_provider"] = provider
            # Streaming responses aren't computationally cached here to keep streaming fast
            async for token in llm.astream(prompt):
                yield token
            return
        except Exception as err:
            print(f"[STREAM FALLBACK] {provider} failed ({err})")
            last_error = err

    raise last_error if last_error else RuntimeError("No LLM provider available")
