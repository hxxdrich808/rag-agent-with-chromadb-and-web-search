# RAG‑Agent with ChromaDB and Web Search

## Requirements
- [high] Vector Store Module: Create `vectorstore.py` exposing `create_vectorstore(persist_directory)` that initializes ChromaDB with OllamaEmbeddings (model="nomic-embed-text") and `load_documents(directory, vectorstore)` that reads .txt/.md files, splits them using RecursiveCharacterTextSplitter, and adds chunks to the collection.
- [high] Agent Tools: Define two LangChain tools via @tool: 
1. `search_local_kb(query, top_k)` – semantic search in ChromaDB.
2. `web_search(query)` – web search using Tavily (requires TAVILY_API_KEY).
- [high] Agent with Routing: Implement `create_agent()` or a LangGraph that routes queries: if the question pertains to local documents, invoke `search_local_kb`; otherwise use `web_search`. The agent’s response must include the source label ("chromadb" or "tavily").
- [normal] CLI Interface: Provide `main.py` with a chat loop that accepts user input until 'exit', calls the agent, and prints both the answer and its source.
- [low] Database Initialization Script: Create a script/command to load test .txt files from a `documents/` directory into ChromaDB using the vector store module before demonstration.
