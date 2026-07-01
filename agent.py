import os
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_executor import ToolExecutorNode
from langgraph.prebuilt.router import RouterNode

from tools import search_local_kb, web_search


def _router(state: Dict[str, Any]) -> str:
    """
    Simple heuristic router that decides whether to use the local KB or web search.
    """
    query = state["messages"][-1]["content"].lower()
    if any(word in query for word in ["news", "current", "today", "recent", "latest"]):
        return "web"
    return "local"


def format_node(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Format the tool output into a final answer and source label.
    """
    if "search_local_kb" in state:
        results = state["search_local_kb"]
        source = "chromadb"
    elif "web_search" in state:
        results = state["web_search"]
        source = "tavily"
    else:
        return {"output": "No result found.", "source": "unknown"}

    # Concatenate the content snippets into a single answer
    answer = "\n\n".join([r["content"] for r in results])
    return {"output": answer, "source": source}


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

    # Formatter node
    formatter_node = format_node

    # Build graph
    graph = StateGraph()
    graph.add_node("router", router_node)
    graph.add_node("local", local_tool)
    graph.add_node("web", web_tool)
    graph.add_node("formatter", formatter_node)

    # Define transitions
    graph.set_entry_point("router")
    graph.add_edge("router", "local")
    graph.add_edge("router", "web")
    graph.add_edge("local", "formatter")
    graph.add_edge("web", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()
