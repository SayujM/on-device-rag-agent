#!/usr/bin/env python3
"""
Launcher for the On-Device RAG Agent with Gradio UI.
"""
import sys
import os

# Initialize environment and module path resolution
import config

from src.agent.rag_agent_with_ui_v3 import RAGInterface
from langgraph.checkpoint.sqlite import SqliteSaver


def main():
    print("=" * 60)
    print("STARTING ON-DEVICE RAG AGENT (Gemma 3 4B + LangGraph)")
    print("=" * 60)
    try:
        with SqliteSaver.from_conn_string(config.CHECKPOINT_DB_PATH) as memory:
            ui = RAGInterface(memory=memory)
            ui.launch()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
