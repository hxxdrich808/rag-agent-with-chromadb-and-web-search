import os

from vectorstore import create_vectorstore, load_documents


def main():
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    docs_dir = os.getenv("DOCS_DIR", "./documents")

    print(f"Loading documents from {docs_dir} into ChromaDB at {persist_dir}")
    vectorstore = create_vectorstore(persist_dir)
    load_documents(docs_dir, vectorstore)
    print("Documents loaded and persisted.")


if __name__ == "__main__":
    main()
