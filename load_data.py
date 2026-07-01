import os
from pathlib import Path

from vectorstore import create_vectorstore, load_documents

def main():
    # Directory containing documents
    docs_dir = Path("documents")
    if not docs_dir.exists():
        print(f"Documents directory '{docs_dir}' does not exist.")
        return

    persist_dir = os.getenv("PERSIST_DIR", "chromadb")
    vectorstore = create_vectorstore(persist_dir)
    load_documents(str(docs_dir), vectorstore)
    print(f"Loaded documents from {docs_dir} into ChromaDB at {persist_dir}")

if __name__ == "__main__":
    main()
