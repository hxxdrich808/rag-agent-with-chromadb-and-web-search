import os
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from vectorstore import create_vectorstore, load_documents
from rag_agent import RAGAgent

# Paths
BASE_DIR = Path(__file__).parent.resolve()
DB_PATH = BASE_DIR / "db"
DOCS_PATH = BASE_DIR / "documents"

console = Console()

def init_vectorstore() -> RAGAgent:
    # Create or load vector store
    vectorstore = create_vectorstore(str(DB_PATH))
    # Load documents if DB is empty
    if not any(DB_PATH.iterdir()):
        console.print("[yellow]Loading documents into ChromaDB...[/yellow]")
        load_documents(str(DOCS_PATH), vectorstore)
    return RAGAgent(vectorstore)

@click.command()
def cli():
    agent = init_vectorstore()
    console.print(Panel("RAG Agent CLI – type 'exit' to quit", title="Welcome"))
    while True:
        try:
            query = click.prompt("[bold cyan]You[/bold cyan]")
            if query.lower() in {"exit", "quit"}:
                break
            result = agent.run(query)
            console.print(Panel(result["output"], title=f"Answer (Source: {result['source']})"))
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    cli()
