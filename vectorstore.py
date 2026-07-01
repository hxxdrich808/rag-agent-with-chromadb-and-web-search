import os
from pathlib import Path
from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def create_vectorstore(persist_directory: str) -> Chroma:
    """
    Create or load a persistent Chroma vector store.

    Args:
        persist_directory (str): Path to the directory where the collection will be stored.

    Returns:
        Chroma: The loaded or newly created vector store.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )


def load_documents(directory: str, vectorstore: Chroma) -> None:
    """
    Load all .txt and .md files from a directory into the provided vector store.

    Args:
        directory (str): Path to the documents folder.
        vectorstore (Chroma): The vector store instance to add chunks to.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs: List[str] = []

    for file_path in Path(directory).rglob("*"):
        if file_path.suffix.lower() in {".txt", ".md"}:
            with open(file_path, "r", encoding="utf-8") as f:
                docs.append(f.read())

    # Split documents into chunks
    chunks = text_splitter.split_text("\n\n".join(docs))
    vectorstore.add_texts(chunks)
    vectorstore.persist()
