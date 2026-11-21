"""Testing fixtures."""

import pytest
from fastapi.testclient import TestClient

from tarachat.app import app
from tarachat.rag import RAGProtocol


class FakeRAGSystem:
    """Minimal RAG pipeline that responds without ML models.

    Implements :class:`~tarachat.rag.RAGProtocol`.
    """

    def add_documents(self, texts, metadatas=None):
        pass

    def retrieve_documents(self, query, k=None):
        return []

    def reset_vector_store(self):
        pass

    def chat(self, message: str, conversation_history: list | None = None):
        yield {"type": "token", "content": f"Echo: {message}"}
        yield {"type": "sources", "sources": ["doc1"]}


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
