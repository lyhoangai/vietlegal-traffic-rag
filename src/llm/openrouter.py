"""src/llm/openrouter.py — OpenRouter async LLM client (fallback)."""
import os
import json
import httpx
from src.llm.base import BaseLLM


class OpenRouterLLM(BaseLLM):
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    async def ainvoke(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://vietlegal-ai-agent.local",
            "X-Title": "VietLegal (Local)",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def astream(self, prompt: str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://vietlegal-ai-agent.local",
            "X-Title": "VietLegal (Local)",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", self.base_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            content = data["choices"][0]["delta"].get("content")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError):
                            pass
