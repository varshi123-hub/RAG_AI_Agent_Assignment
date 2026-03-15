"""
main.py - Entry Point for the RAG AI Agent

Usage:
    1. Place PDF files in the 'data/' folder
    2. Create a .env file with your OPENAI_API_KEY (see .env.example)
    3. Run: python main.py

Flow:
    Phase 1 - Ingestion:
        PDF files -> Extract text -> Split into chunks -> Store in ChromaDB

    Phase 2 - RAG Agent (Interactive Q&A):
        User question -> Retrieve relevant chunks -> LLM generates answer -> Response
"""

import os
from dotenv import load_dotenv

# Load environment variables (OPENAI_API_KEY) from .env file
load_dotenv()

from ingestion import run_ingestion, CHROMA_DB_DIR
from rag_agent import query_rag


def main():
    print("=" * 60)
    print("       RAG AI Agent - LangGraph + ChromaDB")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Phase 1: Ingestion (only if vector DB doesn't exist yet)
    # ------------------------------------------------------------------
    if not os.path.exists(CHROMA_DB_DIR):
        print("No existing vector database found. Running ingestion...\n")
        result = run_ingestion()
        if result is None:
            print("\nIngestion failed. Please add PDFs to 'data/' and try again.")
            return
        print()
    else:
        print(f"Found existing vector database at '{CHROMA_DB_DIR}/'.")
        print("Skipping ingestion. Delete the folder to re-ingest.\n")

    # ------------------------------------------------------------------
    # Phase 2: RAG Agent - Interactive Q&A Loop
    # ------------------------------------------------------------------
    print("=" * 60)
    print("       RAG Agent Ready - Ask Questions!")
    print("=" * 60)
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()
        # Run the question through the LangGraph RAG agent
        answer = query_rag(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
