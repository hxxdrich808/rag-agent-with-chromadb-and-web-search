import os
from pathlib import Path
from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def create_vectorstore(persist_directory: str) -> Chroma:
    """
    Create or load a Chroma vector store persisted at the given directory.
    Uses Ollama embeddings with model 'nomic-embed-text'.
    """
    if not os.path.isdir(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def load_documents(directory: str, vectorstore: Chroma) -> None:
    """
    Load all .txt and .md files from the given directory into the provided vector store.
    The documents are split into chunks using RecursiveCharacterTextSplitter before being added.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs: List[str] = []

    for file_path in Path(directory).rglob("*"):
        if file_path.suffix.lower() in {".txt", ".md"}:
            try:
                text = file_path.read_text(encoding="utf-8")
                docs.append(text)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

    # Split documents into chunks
    chunks = splitter.split_documents([{"page_content": d} for d in docs])

    # Prepare metadata and add to vector store
    metadatas = [{"source": str(file_path)} for file_path in Path(directory).rglob("*") if file_path.suffix.lower() in {".txt", ".md"}]
    # Since we split multiple documents, replicate metadata accordingly
    # For simplicity, assign generic source "local_kb"
    vectorstore.add_texts(
        [chunk["page_content"] for chunk in chunks],
        metadatas=[{"source": "local_kb"}] * len(chunks)
    )
    vectorstore.persist()
