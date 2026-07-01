# RAG‑Agent with ChromaDB and Web Search

## Requirements
- [high] Use ChromaDB instead of Qdrant: Replace any Qdrant usage with ChromaDB. Ensure vectorstore.py creates a Chroma collection using OllamaEmbeddings (model="nomic-embed-text") and persists to disk at ./chroma_db.
- [high] Install required packages: Add langchain-chroma, langchain-ollama, tavily-python to the project dependencies. Verify that pip install works with these packages and that they are imported correctly in code.
- [high] Implement vectorstore module: Create vectorstore.py exposing create_vectorstore(persist_directory) and load_documents(directory, vectorstore). Use RecursiveCharacterTextSplitter for .txt/.md files and add chunks to the Chroma collection.
- [high] Define agent tools with @tool decorators: Implement search_local_kb(query, top_k) performing semantic search on ChromaDB via retriever. Implement web_search(query) using Tavily API (TAVILY_API_KEY from .env). Both should return results and indicate source.
- [high] Create routing agent: Build create_agent() or a LangGraph that selects between local KB search and web search based on query context. The response must include the source label (chromadb / tavily). Use system prompt to guide selection.
- [normal] CLI chat loop: Implement main.py with a simple REPL: read user input, pass to agent, print answer and source. Exit on 'exit'. Ensure graceful handling of errors.
- [normal] Data loading script: Provide a script or command that loads all .txt/.md files from documents/ into ChromaDB using load_documents before starting the agent. Persist data so it survives restarts.
