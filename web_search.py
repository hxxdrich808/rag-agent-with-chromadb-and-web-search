import os
from typing import List, Dict
import tavily

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def tavily_search(query: str, max_results: int = 3) -> str:
    """
    Query Tavily API and return concatenated results as a single string.
    """
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY is missing in environment")
    client = tavily.Client(api_key=TAVILY_API_KEY)
    response: Dict[str, List[Dict]] = client.search(query=query, max_results=max_results)
    results = []
    for item in response.get("results", []):
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        url = item.get("url", "")
        results.append(f"{title}\n{snippet}\nURL: {url}")
    return "\n\n".join(results)
