import os
from pathlib import Path
from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_vectorstore(persist_directory: str) -> Chroma:
    """
    Create or load a Chroma vector store using Ollama embeddings.
    """
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )


def load_documents(directory: str, vectorstore: Chroma) -> None:
    """
    Load .txt and .md files from `directory`, split them, and add to the vector store.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    for file_path in Path(directory).rglob("*"):
        if file_path.suffix.lower() in {".txt", ".md"}:
            text = file_path.read_text(encoding="utf-8")
            docs.extend(splitter.split_text(text))
    vectorstore.add_texts(docs)
    vectorstore.persist()
