import os
from typing import List, Dict

from langchain_chroma import Chroma
from tavily_python import TavilyClient


def search_local_kb(query: str, top_k: int = 3) -> List[Dict]:
    """
    Perform a semantic search on the local ChromaDB collection.

    Returns:
        List of dictionaries with 'content' and 'source' keys.
    """
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    vectorstore = Chroma(persist_directory=persist_dir)
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    return [
        {"content": doc.page_content, "source": f"chromadb_{i+1}"}
        for i, (doc, _) in enumerate(results)
    ]


def web_search(query: str) -> List[Dict]:
    """
    Perform a web search using Tavily.

    Returns:
        List of dictionaries with 'content' and 'source' keys.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")

    client = TavilyClient(api_key=api_key)
    results = client.search(query, max_results=3)

    return [
        {"content": r["content"], "source": f"tavily_{i+1}"}
        for i, r in enumerate(results)
    ]
