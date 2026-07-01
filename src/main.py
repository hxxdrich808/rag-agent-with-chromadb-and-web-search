import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .vectorstore import create_vectorstore, load_documents
from .agent import create_agent

def main():
    load_dotenv()
    # Ensure API keys are set
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY is not set. Please add it to your .env file.")
        sys.exit(1)

    persist_dir = "./chroma_db"
    vectorstore = create_vectorstore(persist_dir)

    # Load documents if the database is empty
    if not Path(persist_dir).exists() or len(vectorstore.get_index().get_all_documents()) == 0:
        print("Loading local documents into ChromaDB...")
        load_documents("./documents", vectorstore)
        print("Documents loaded and persisted.")

    agent = create_agent(vectorstore)

    print("\n=== RAG-Agent CLI ===")
    print("Type your question (or 'exit' to quit).")

    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            if not user_input:
                continue

            answer, sources = agent(user_input)
            print(f"\nAnswer:\n{answer}")
            if sources:
                print(f"Sources: {', '.join(sources)}")
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
