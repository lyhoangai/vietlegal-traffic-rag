"""tests/test_llm.py — LLM fallback unit tests."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_invoke_uses_gemini_first():
    from src.llm import invoke_with_fallback
    state = {"llm_provider": "gemini"}
    with patch("src.llm.GeminiLLM") as MockGemini:
        MockGemini.return_value.ainvoke = AsyncMock(return_value="câu trả lời test")
        result = await invoke_with_fallback("câu hỏi test", state)
    assert result == "câu trả lời test"
    assert state["llm_provider"] == "gemini"


@pytest.mark.asyncio
async def test_fallback_to_groq_on_gemini_error():
    from src.llm import invoke_with_fallback
    state = {"llm_provider": "gemini"}
    with patch("src.llm.GeminiLLM") as MockGemini, \
         patch("src.llm.GroqLLM") as MockGroq:
        MockGemini.return_value.ainvoke = AsyncMock(side_effect=Exception("rate limit"))
        MockGroq.return_value.ainvoke = AsyncMock(return_value="groq answer")
        result = await invoke_with_fallback("test", state)
    assert result == "groq answer"
    assert state["llm_provider"] == "groq"
