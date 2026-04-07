import pytest
from pydantic import ValidationError

from tarachat.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DocumentUpload,
    HealthResponse,
)


class TestChatMessage:
    def test_valid_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_role(self):
        msg = ChatMessage(role="assistant", content="Hi there")
        assert msg.role == "assistant"

    def test_missing_role_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(content="Hello")

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user")


class TestChatRequest:
    def test_valid_request(self):
        req = ChatRequest(message="Hello")
        assert req.message == "Hello"
        assert req.conversation_history == []

    def test_with_history(self):
        history = [{"role": "user", "content": "Hi"}]
        req = ChatRequest(message="Hello", conversation_history=history)
        assert len(req.conversation_history) == 1

    def test_empty_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_missing_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest()


class TestChatResponse:
    def test_valid_response(self):
        resp = ChatResponse(response="Hello!", sources=["doc1"])
        assert resp.response == "Hello!"
        assert resp.sources == ["doc1"]

    def test_default_sources(self):
        resp = ChatResponse(response="Hello!")
        assert resp.sources == []


class TestDocumentUpload:
    def test_valid_upload(self):
        doc = DocumentUpload(content="Some text")
        assert doc.content == "Some text"
        assert doc.metadata == {}

    def test_with_metadata(self):
        doc = DocumentUpload(content="Text", metadata={"author": "John"})
        assert doc.metadata["author"] == "John"


class TestHealthResponse:
    def test_healthy(self):
        resp = HealthResponse(status="healthy", model_loaded=True, vector_store_ready=True)
        assert resp.status == "healthy"
        assert resp.model_loaded is True

    def test_initializing(self):
        resp = HealthResponse(status="initializing", model_loaded=False, vector_store_ready=False)
        assert resp.status == "initializing"
