"""src/agent/state.py — Full AgentState for senior-level architecture."""
from typing import TypedDict, Optional, Annotated
import operator


class AgentState(TypedDict):
    # Conversation history (for memory)
    messages: Annotated[list, operator.add]

    # Current query
    user_query: str

    # Intent classification result ("penalty" | "law" | "speed" | "general")
    intent: str

    # Entities extracted by LLM (vehicle_type, violation_type, etc.)
    entities: dict

    # Retrieved docs (before and after reranking)
    retrieved_docs: list
    reranked_docs: list
    web_docs: list

    # Sources cited
    sources: list

    # Human-in-the-loop clarification
    needs_clarification: bool
    clarification_question: str

    # Output
    answer: str
    confidence: float

    # Routing metadata
    llm_provider: str
    collection_used: str
