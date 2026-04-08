"""Testing fixtures."""

import json

import pytest
from fastapi.testclient import TestClient

from tarachat.app import app
from tarachat.rag import RAGProtocol


class FakeRAGSystem:
    """Minimal RAG system that responds without ML models.

    Implements :class:`~tarachat.rag.RAGProtocol`.
    """

    def __init__(self):
        self.model = object()
        self.vector_store = object()

    def add_documents(self, texts, metadatas=None):
        pass

    def retrieve_documents(self, query, k=None):
        return []

    def create_empty_vector_store(self):
        return object()

    def chat(self, message: str, conversation_history: list | None = None):
        answer = f"Echo: {message}"
        yield f'data: {json.dumps({"type": "token", "content": answer})}\n\n'
        yield f'data: {json.dumps({"type": "sources", "sources": ["doc1"]})}\n\n'
        yield "data: [DONE]\n\n"


assert isinstance(FakeRAGSystem(), RAGProtocol), "FakeRAGSystem must satisfy RAGProtocol"


@pytest.fixture(scope="module")
def fake_rag():
    """Fake RAG system fixture."""
    return FakeRAGSystem()


@pytest.fixture(scope="module")
def client(fake_rag):
    """Test client with fake RAG system."""
    app.state.rag = fake_rag
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    del app.state.rag
