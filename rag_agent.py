import os
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

from vectorstore import Chroma  # type: ignore
from web_search import tavily_search

load_dotenv()

# Global environment variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------- Tools ----------
@tool
def search_local_kb(query: str, top_k: int = 3) -> str:
    """
    Perform a semantic search in the local ChromaDB vector store.
    Returns concatenated top-k documents as context string.
    """
    global VECTORSTORE
    results = VECTORSTORE.similarity_search_with_score(query, k=top_k)
    contexts = [res.page_content for res, _ in results]
    return "\n\n".join(contexts)


@tool
def web_search(query: str) -> str:
    """
    Perform a web search using Tavily API.
    Returns concatenated top results as context string.
    """
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY is not set in .env")
    return tavily_search(query)


# ---------- Agent ----------
class RAGAgent:
    def __init__(self, vectorstore: Chroma):
        global VECTORSTORE
        VECTORSTORE = vectorstore

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY,
        )

        # Build the graph
        self.graph = StateGraph()
        self.graph.add_node("agent", create_react_agent(self.llm, [search_local_kb, web_search]))
        self.graph.set_entry_point("agent")
        self.graph.add_edge("agent", END)
        self.app = self.graph.compile()

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent on a user query.
        Returns dict with 'output' and 'source'.
        """
        result = self.app.invoke({"messages": [{"role": "user", "content": query}]})
        # Extract source from tool calls
        tools_used = [msg["name"] for msg in result.get("messages", []) if msg.get("role") == "tool"]
        source = "chromadb" if "search_local_kb" in tools_used else "tavily"
        return {"output": result.get("messages")[-1]["content"], "source": source}
