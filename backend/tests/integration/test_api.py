"""Integration tests for the API service."""

import json


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE data lines into a list of JSON objects."""
    events = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
    return events


def test_root(client):
    """The API should expose a root endpoint with navigation links."""
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Welcome to TaraChat API"
    assert "docs" in body
    assert "health" in body


def test_health(client):
    """The API should report healthy when the RAG system is ready."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["vector_store_ready"] is True


def test_health_initializing(client, fake_rag):
    """The API should report initializing when the RAG system is not ready."""
    fake_rag._ready = False
    try:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "initializing"
    finally:
        fake_rag._ready = True


def test_chat(client):
    """The API should stream a chat response via SSE."""
    response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = _parse_sse(response.text)
    tokens = [e for e in events if e["type"] == "token"]
    sources = [e for e in events if e["type"] == "sources"]
    assert len(tokens) >= 1
    assert tokens[0]["content"] == "Echo: Hello"
    assert len(sources) == 1
    assert "doc1" in sources[0]["sources"]


def test_chat_with_history(client):
    """The API should accept conversation history in chat requests."""
    response = client.post("/chat", json={
        "message": "Follow up",
        "conversation_history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ],
    })

    assert response.status_code == 200
    events = _parse_sse(response.text)
    tokens = [e for e in events if e["type"] == "token"]
    assert tokens[0]["content"] == "Echo: Follow up"


def test_chat_not_ready(client, fake_rag):
    """The API should return 503 when the RAG system is not ready."""
    fake_rag._ready = False
    try:
        response = client.post("/chat", json={"message": "Hello"})

        assert response.status_code == 503
    finally:
        fake_rag._ready = True


def test_chat_empty_message(client):
    """The API should reject empty messages with 422."""
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422


def test_chat_missing_message(client):
    """The API should reject missing message field with 422."""
    response = client.post("/chat", json={})

    assert response.status_code == 422


def test_cors_headers(client):
    """The API should include CORS headers for allowed origins."""
    response = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_openapi_docs(client):
    """The API should expose OpenAPI documentation."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "TaraChat API"
    assert "/chat" in schema["paths"]
    assert "/health" in schema["paths"]
