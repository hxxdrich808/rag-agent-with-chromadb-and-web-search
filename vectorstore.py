import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def create_vectorstore(persist_directory: str) -> Chroma:
    """
    Create or load a persistent Chroma vector store using Ollama embeddings.
    """
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)

    embedding = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
    )


def load_documents(directory: str, vectorstore: Chroma) -> None:
    """
    Load all .txt and .md files from the given directory into the provided vector store.
    The documents are split using RecursiveCharacterTextSplitter before being added.
    """
    docs = []
    for file_path in Path(directory).rglob("*"):
        if file_path.suffix.lower() in {".txt", ".md"}:
            loader = TextLoader(str(file_path))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Add to vector store
    vectorstore.add_documents(chunks)
