import os
from pathlib import Path

import pytest

from vectorstore import create_vectorstore, load_documents
from rag_agent import RAGAgent

BASE_DIR = Path(__file__).parent.parent.resolve()
DB_PATH = BASE_DIR / "db"
DOCS_PATH = BASE_DIR / "documents"

@pytest.fixture(scope="module")
def agent():
    # Ensure DB exists with sample docs
    vectorstore = create_vectorstore(str(DB_PATH))
    if not any(DB_PATH.iterdir()):
        load_documents(str(DOCS_PATH), vectorstore)
    return RAGAgent(vectorstore)

def test_local_search(agent):
    res = agent.run("What is the capital of France?")
    assert "Paris" in res["output"]
    assert res["source"] == "chromadb"

def test_web_fallback(agent, monkeypatch):
    # Simulate no local docs by clearing DB
    vectorstore = create_vectorstore(str(DB_PATH))
    vectorstore.delete_collection()
    agent_local = RAGAgent(vectorstore)
    res = agent_local.run("Who won the 2024 US presidential election?")
    assert "tavily" in res["source"] or "Answer" in res["output"]
