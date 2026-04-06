#!/usr/bin/env python3
"""
Script to initialize the vector store with sample documents.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag import rag_system


def main():
    """Initialize the RAG system with sample data."""
    print("Initializing RAG system...")
    rag_system.initialize()

    # Load sample documents
    data_path = Path(__file__).parent.parent / "data" / "sample_documents.txt"

    if not data_path.exists():
        print(f"Warning: Sample documents not found at {data_path}")
        print("Vector store initialized but empty.")
        return

    print(f"Loading documents from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into paragraphs
    documents = [doc.strip() for doc in content.split("\n\n") if doc.strip()]

    print(f"Adding {len(documents)} documents to vector store...")
    rag_system.add_documents(documents)

    print("✓ Initialization complete!")
    print(f"Vector store contains {rag_system.vector_store.index.ntotal} embeddings")


if __name__ == "__main__":
    main()
