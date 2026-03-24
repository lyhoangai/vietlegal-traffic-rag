def _make_state(query: str) -> dict:
    return {
        "messages": [],
        "user_query": query,
        "intent": "general",
        "entities": {},
        "retrieved_docs": [],
        "reranked_docs": [],
        "web_docs": [],
        "sources": [],
        "needs_clarification": False,
        "clarification_question": "",
        "answer": "",
        "confidence": 0.0,
        "llm_provider": "gemini",
        "collection_used": "traffic_law",
    }
