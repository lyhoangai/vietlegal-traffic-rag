"""src/llm/gemini.py — Gemini async LLM client with real streaming."""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from src.llm.base import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self):
        self.client = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    async def ainvoke(self, prompt: str) -> str:
        response = await self.client.ainvoke(prompt)
        return response.content

    async def astream(self, prompt: str):
        """Real token-by-token streaming from Gemini."""
        async for chunk in self.client.astream(prompt):
            if chunk.content:
                yield chunk.content
