"""Testing fixtures."""

import json

import pytest
from fastapi.testclient import TestClient

from tarachat.main import app, get_rag_system


class FakeRAGSystem:
    """Minimal RAG system that responds without ML models."""

    def __init__(self):
        self.model = object()
        self.vector_store = object()
        self._ready = True

    def initialize(self):
        pass

    def is_ready(self) -> bool:
        return self._ready

    def chat_stream(self, message: str, history: list | None = None):
        answer = f"Echo: {message}"
        yield f'data: {json.dumps({"type": "token", "content": answer})}\n\n'
        yield f'data: {json.dumps({"type": "sources", "sources": ["doc1"]})}\n\n'
        yield "data: [DONE]\n\n"


@pytest.fixture(scope="module")
def fake_rag():
    """Fake RAG system fixture."""
    return FakeRAGSystem()


@pytest.fixture(scope="module")
def client(fake_rag):
    """Test client with fake RAG system."""
    app.dependency_overrides[get_rag_system] = lambda: fake_rag
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()
