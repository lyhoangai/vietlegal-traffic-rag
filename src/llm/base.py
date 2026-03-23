"""src/llm/base.py — Abstract LLM interface."""
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    async def ainvoke(self, prompt: str) -> str:
        """Invoke the LLM and return the full response."""
        pass

    @abstractmethod
    async def astream(self, prompt: str):
        """Async generator yielding str tokens."""
        pass
