"""src/llm/__init__.py — LLM factory with Gemini → Groq fallback."""
import os
from src.llm.gemini import GeminiLLM
from src.llm.groq_client import GroqLLM


def get_llm(provider: str = None):
    """Return an LLM by provider name."""
    provider = provider or os.getenv("LLM_PROVIDER", "gemini")
    if provider == "groq":
        return GroqLLM()
    return GeminiLLM()


async def invoke_with_fallback(prompt: str, state: dict) -> str:
    """Call preferred provider first; then fallback to the other provider."""
    preferred = (state or {}).get("llm_provider") or os.getenv("LLM_PROVIDER", "gemini")
    order = ["groq", "gemini"] if preferred == "groq" else ["gemini", "groq"]
    last_error = None

    for provider in order:
        try:
            llm = GroqLLM() if provider == "groq" else GeminiLLM()
            result = await llm.ainvoke(prompt)
            state["llm_provider"] = provider
            return result
        except Exception as err:
            print(f"[LLM FALLBACK] {provider} failed ({err})")
            last_error = err

    raise last_error if last_error else RuntimeError("No LLM provider available")


async def stream_with_fallback(prompt: str, state: dict):
    """Stream with preferred provider first; fallback to the other provider."""
    preferred = (state or {}).get("llm_provider") or os.getenv("LLM_PROVIDER", "gemini")
    order = ["groq", "gemini"] if preferred == "groq" else ["gemini", "groq"]
    last_error = None

    for provider in order:
        try:
            llm = GroqLLM() if provider == "groq" else GeminiLLM()
            state["llm_provider"] = provider
            async for token in llm.astream(prompt):
                yield token
            return
        except Exception as err:
            print(f"[STREAM FALLBACK] {provider} failed ({err})")
            last_error = err

    raise last_error if last_error else RuntimeError("No LLM provider available")
