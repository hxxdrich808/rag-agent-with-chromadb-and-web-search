import os
from typing import List, Dict

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI  # For demonstration; replace with your LLM if needed
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from .tools import search_local_kb, web_search

def create_agent(vectorstore: Chroma):
    """
    Creates a simple routing agent that decides whether to use local KB or web search.
    The decision is based on the presence of certain keywords in the query.
    Returns a callable that accepts a user query and returns answer + source list.
    """

    # Define tool functions
    def retrieve_local(query: str, top_k: int = 3):
        return search_local_kb(vectorstore, query, top_k)

    def browse_web(query: str, top_k: int = 3):
        return web_search(query, top_k)

    tools = {
        "retrieve_local": retrieve_local,
        "browse_web": browse_web
    }

    # Simple routing logic based on keywords
    def router(state):
        query = state["messages"][-1]["content"]
        if any(word in query.lower() for word in ["news", "latest", "current", "today"]):
            return "web"
        else:
            return "local"

    workflow = StateGraph()
    workflow.add_node("router", router)
    workflow.add_node("local", create_react_agent(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        tools=[retrieve_local],
        system_message="You are a helpful assistant that answers questions using the local knowledge base."
    ))
    workflow.add_node("web", create_react_agent(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        tools=[browse_web],
        system_message="You are a helpful assistant that answers questions by searching the web."
    ))

    workflow.set_entry_point("router")
    workflow.add_edge("router", "local")
    workflow.add_edge("router", "web")
    workflow.add_edge("local", "__end__")
    workflow.add_edge("web", "__end__")

    graph = workflow.compile()

    def agent(query: str):
        result = graph.invoke({"messages": [{"role": "user", "content": query}]})
        # Extract sources from tool calls
        sources = []
        for msg in result["messages"]:
            if msg.get("tool_calls"):
                for call in msg["tool_calls"]:
                    sources.append(call.get("name"))
        return result["messages"][-1]["content"], sources

    return agent
