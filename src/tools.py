import os
from typing import List, Dict

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in the environment")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def search_local_kb(vectorstore: Chroma, query: str, top_k: int = 3) -> List[Dict]:
    """
    Perform a semantic search on the local ChromaDB vector store.
    Returns a list of dictionaries containing 'content' and 'source'.
    """
    results = vectorstore.similarity_search(query, k=top_k)
    return [{"content": r.page_content, "source": r.metadata.get("source", "local_kb")} for r in results]

def web_search(query: str, top_k: int = 3) -> List[Dict]:
    """
    Perform a web search using Tavily API.
    Returns a list of dictionaries containing 'content' and 'source'.
    """
    response = tavily_client.search(
        query=query,
        max_results=top_k
    )
    results = []
    for hit in response.get("results", []):
        content = hit.get("snippet") or ""
        url = hit.get("url")
        results.append({"content": content, "source": f"tavily:{url}"})
    return results
