import os
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_executor import ToolExecutorNode
from langgraph.prebuilt.router import RouterNode
from langgraph.prebuilt.tools import create_tool_node

from tools import search_local_kb, web_search


def _router(state: Dict[str, Any]) -> str:
    """
    Simple heuristic router that decides whether to use the local KB or web search.
    """
    query = state["messages"][-1]["content"].lower()
    if any(word in query for word in ["news", "current", "today", "recent", "latest"]):
        return "web"
    return "local"


def create_agent() -> StateGraph:
    """
    Build and return a LangGraph agent that routes queries to the appropriate tool.
    The output includes the answer text and the source label.
    """
    # Define tools
    local_tool = ToolExecutorNode(
        {"search_local_kb": search_local_kb},
        name="local",
    )
    web_tool = ToolExecutorNode(
        {"web_search": web_search},
        name="web",
    )

    # Router node
    router_node = RouterNode(_router)

    # Build graph
    graph = StateGraph()
    graph.add_node("router", router_node)
    graph.add_node("local", local_tool)
    graph.add_node("web", web_tool)

    # Define transitions
    graph.set_entry_point("router")
    graph.add_edge("router", "local")
    graph.add_edge("router", "web")
    graph.add_edge("local", END)
    graph.add_edge("web", END)

    return graph.compile()
