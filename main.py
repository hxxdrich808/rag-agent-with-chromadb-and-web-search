import os
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel
from langgraph.graph import StateGraph

from agent import create_agent


def main():
    console = Console()
    agent = create_agent()

    console.print("[bold green]RAG Agent is ready![/bold green]")
    console.print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        # Build the initial state
        state = {"messages": [{"role": "user", "content": user_input}]}

        # Run the agent
        result = agent.invoke(state)

        # Extract answer and source
        if isinstance(result, dict):
            answer = result.get("output")
            source = result.get("source", "unknown")
        else:
            answer = str(result)
            source = "unknown"

        console.print(
            Panel(
                f"[bold]Answer:[/bold]\n{answer}\n\n[italic]Source: {source}[/italic]",
                title="Response",
            )
        )


if __name__ == "__main__":
    main()
