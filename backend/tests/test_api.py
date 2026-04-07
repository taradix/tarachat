import json

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.main import app, get_rag_system
from app.rag import RAGSystem


@pytest.fixture
def mock_rag():
    """Create a mock RAG system for testing."""
    rag = MagicMock(spec=RAGSystem)
    rag.is_ready.return_value = True
    rag.model = MagicMock()
    rag.vector_store = MagicMock()
    rag.chat.return_value = ("Test response", ["source1"])

    def fake_stream(message, history=None):
        yield f'data: {json.dumps({"type": "token", "content": "Test response"})}\n\n'
        yield f'data: {json.dumps({"type": "sources", "sources": ["source1"]})}\n\n'
        yield "data: [DONE]\n\n"

    rag.chat_stream.side_effect = fake_stream
    return rag


@pytest.fixture
def client(mock_rag):
    """Create a test client with injected mock RAG system."""
    app.dependency_overrides[get_rag_system] = lambda: mock_rag
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestRootEndpoint:
    def test_root_returns_welcome(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Welcome to TaraChat API"
        assert "docs" in data
        assert "health" in data


class TestHealthEndpoint:
    def test_healthy_status(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["vector_store_ready"] is True

    def test_initializing_status(self, client, mock_rag):
        mock_rag.is_ready.return_value = False
        mock_rag.model = None
        mock_rag.vector_store = None
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "initializing"
        assert data["model_loaded"] is False


def _parse_sse_events(response_text: str) -> list:
    """Parse SSE events from response text."""
    events = []
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if line.startswith('data: ') and line != 'data: [DONE]':
            events.append(json.loads(line[6:]))
    return events


class TestChatEndpoint:
    def test_successful_chat_stream(self, client, mock_rag):
        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = _parse_sse_events(response.text)
        token_events = [e for e in events if e["type"] == "token"]
        source_events = [e for e in events if e["type"] == "sources"]
        assert len(token_events) >= 1
        assert token_events[0]["content"] == "Test response"
        assert len(source_events) == 1
        assert source_events[0]["sources"] == ["source1"]

    def test_chat_with_history(self, client, mock_rag):
        payload = {
            "message": "Follow up",
            "conversation_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ]
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 200

    def test_chat_empty_message_returns_422(self, client):
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422

    def test_chat_missing_message_returns_422(self, client):
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_when_not_ready_returns_503(self, client, mock_rag):
        mock_rag.is_ready.return_value = False
        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 503
