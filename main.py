import os
from rich.console import Console
from rich.markdown import Markdown

from agent import create_agent

console = Console()

def main():
    console.print("[bold cyan]Welcome to the RAG Agent![/]")
    agent = create_agent()
    while True:
        try:
            user_input = input("\n[green]You:[/ ] ")
            if user_input.lower() in {"exit", "quit"}:
                console.print("[bold red]Goodbye![/]")
                break
            answer, source = agent(user_input)
            console.print(Markdown(f"**Answer ({source}):**\n{answer}"))
        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted. Exiting.[/]")
            break

if __name__ == "__main__":
    main()
