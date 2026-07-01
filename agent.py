import os
from typing import Tuple, Dict, Any

from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain

# Load environment variables for Tavily API key
from dotenv import load_dotenv

load_dotenv()


def create_agent() -> Any:
    """
    Create a simple routing agent that decides whether to use local KB or web search.
    Returns a callable that takes a user query and returns (answer, source).
    """

    llm = Ollama(model="llama3.1")

    # Prompt instructing the LLM to choose a tool
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant that decides whether to answer a question using local documents or by searching the web."),
            (
                "user",
                "Question: {question}\n"
                "Respond with either 'local' or 'web'. Do not add any other text.",
            ),
        ]
    )

    chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())

    def agent(query: str) -> Tuple[str, str]:
        # Decide tool
        decision = chain.invoke({"question": query}).strip().lower()
        if decision == "local":
            results = search_local_kb(query)
            source = "chromadb"
            answer = summarize_results(results)
        else:
            results = web_search(query)
            source = "tavily"
            answer = summarize_results(results)

        return answer, source

    return agent


def summarize_results(results: list) -> str:
    """
    Simple summarizer that concatenates the first few result contents.
    """
    if not results:
        return "I couldn't find any relevant information."
    snippets = []
    for res in results[:3]:
        content = res.get("content") or res.get("title", "")
        snippets.append(content)
    return "\n\n".join(snippets)


# Import tools after defining agent to avoid circular imports
from tools import search_local_kb, web_search
