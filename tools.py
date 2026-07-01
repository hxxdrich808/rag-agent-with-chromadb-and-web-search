import os
from typing import List, Dict

from langchain.tools import tool
from langchain_tavily import TavilySearchResults
from chromadb.api.types import Documents


@tool("search_local_kb")
def search_local_kb(query: str, top_k: int = 3) -> List[Dict]:
    """
    Perform a semantic search on the local ChromaDB collection.
    Returns a list of dictionaries containing 'content' and 'source'.
    """
    from vectorstore import create_vectorstore

    # Load persistent store
    vectorstore = create_vectorstore(os.getenv("PERSIST_DIR", "chromadb"))
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs: Documents = retriever.get_relevant_documents(query)

    results = []
    for doc in docs:
        results.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
            }
        )
    return results


@tool("web_search")
def web_search(query: str) -> List[Dict]:
    """
    Perform a web search using Tavily and return the top 3 results.
    Each result contains 'title', 'url', and 'content'.
    """
    tavily = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
    raw_results = tavily.run(query, max_results=3)

    results = []
    for res in raw_results:
        results.append(
            {
                "title": res.get("title", ""),
                "url": res.get("link", ""),
                "content": res.get("snippet", ""),
            }
        )
    return results
