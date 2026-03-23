"""LangGraph state machine for the VietLegal chat pipeline."""

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    clarifier,
    generator,
    intent_analyzer,
    query_router,
    reranker,
    retriever,
    web_searcher,
)
from src.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent_analyzer", intent_analyzer)
    graph.add_node("clarifier", clarifier)
    graph.add_node("query_router", query_router)
    graph.add_node("retriever", retriever)
    graph.add_node("reranker", reranker)
    graph.add_node("web_searcher", web_searcher)
    graph.add_node("generator", generator)

    graph.set_entry_point("intent_analyzer")

    graph.add_conditional_edges(
        "intent_analyzer",
        lambda state: "clarifier" if state.get("needs_clarification") else "query_router",
        {"clarifier": "clarifier", "query_router": "query_router"},
    )
    graph.add_edge("clarifier", END)
    graph.add_edge("query_router", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "web_searcher")
    graph.add_edge("web_searcher", "generator")
    graph.add_edge("generator", END)

    return graph.compile()


agent = build_graph()
