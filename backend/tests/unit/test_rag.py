"""Unit tests for RAGSystem pure methods (no ML models required)."""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from tarachat.config import Settings
from tarachat.rag import RAGSystem


@pytest.fixture
def rag(tmp_path):
    """RAGSystem with mock ML components."""
    settings = Settings(vector_store_path=str(tmp_path / "vs"))
    return RAGSystem(
        settings=settings,
        device="cpu",
        embeddings=MagicMock(),
        vector_store=MagicMock(),
        tokenizer=MagicMock(),
        model=MagicMock(),
    )


class TestBuildPrompt:
    def test_basic_prompt(self, rag):
        docs = [Document(page_content="Some context")]
        result = rag._build_prompt("What is X?", docs)
        assert "Some context" in result
        assert "Question : What is X?" in result
        assert "Réponse :" in result

    def test_prompt_without_history(self, rag):
        docs = [Document(page_content="ctx")]
        result = rag._build_prompt("Q?", docs)
        assert "Historique" not in result

    def test_prompt_with_history(self, rag):
        docs = [Document(page_content="ctx")]
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = rag._build_prompt("Q?", docs, history)
        assert "Historique de la conversation" in result
        assert "Utilisateur: Hi" in result
        assert "Assistant: Hello" in result

    def test_prompt_truncates_history_to_6(self, rag):
        docs = [Document(page_content="ctx")]
        history = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        result = rag._build_prompt("Q?", docs, history)
        # Only last 6 should appear
        assert "msg4" in result
        assert "msg9" in result
        assert "msg3" not in result

    def test_multiple_docs_joined(self, rag):
        docs = [
            Document(page_content="First doc"),
            Document(page_content="Second doc"),
        ]
        result = rag._build_prompt("Q?", docs)
        assert "First doc" in result
        assert "Second doc" in result


class TestBuildDemoResponse:
    def test_with_docs(self, rag):
        docs = [Document(page_content="Important content here")]
        result = rag._build_demo_response(docs)
        assert "Important content here" in result
        assert "trouvé" in result

    def test_with_two_docs(self, rag):
        docs = [
            Document(page_content="First"),
            Document(page_content="Second"),
        ]
        result = rag._build_demo_response(docs)
        assert "First" in result
        assert "Second" in result

    def test_no_docs(self, rag):
        result = rag._build_demo_response([])
        assert "Désolé" in result

    def test_long_doc_truncated(self, rag):
        docs = [Document(page_content="x" * 500)]
        result = rag._build_demo_response(docs)
        assert len(result) < 500


class TestExtractSources:
    def test_source_preview(self, rag):
        docs = [Document(page_content="Hello world from document")]
        sources = rag._extract_sources(docs)
        assert len(sources) == 1
        assert sources[0].endswith("...")

    def test_multiple_sources(self, rag):
        docs = [
            Document(page_content="Doc A"),
            Document(page_content="Doc B"),
        ]
        sources = rag._extract_sources(docs)
        assert len(sources) == 2

    def test_long_source_truncated(self, rag):
        docs = [Document(page_content="x" * 200)]
        sources = rag._extract_sources(docs)
        assert len(sources[0]) == 103  # 100 chars + "..."


class TestRetrieveDocuments:
    def test_empty_index_returns_empty(self, rag):
        rag.vector_store.index.ntotal = 0
        assert rag.retrieve_documents("query") == []

    def test_uses_settings_top_k(self, rag):
        rag.vector_store.index.ntotal = 5
        rag.vector_store.similarity_search.return_value = [Document(page_content="hit")]
        result = rag.retrieve_documents("query")
        rag.vector_store.similarity_search.assert_called_once_with("query", k=rag.settings.top_k)
        assert len(result) == 1

    def test_custom_k(self, rag):
        rag.vector_store.index.ntotal = 5
        rag.vector_store.similarity_search.return_value = []
        rag.retrieve_documents("query", k=7)
        rag.vector_store.similarity_search.assert_called_once_with("query", k=7)


class TestAddDocuments:
    def test_empty_texts_is_noop(self, rag):
        rag.add_documents([])
        # No error, no vector store interaction

    def test_adds_and_saves(self, rag, tmp_path):
        rag.settings.vector_store_path = str(tmp_path / "vs")
        rag.settings.chunk_size = 5000  # large enough to avoid splitting
        rag.add_documents(["hello world"], [{"source": "test"}])
        rag.vector_store.add_documents.assert_called_once()
        docs = rag.vector_store.add_documents.call_args[0][0]
        assert any("hello world" in d.page_content for d in docs)
        rag.vector_store.save_local.assert_called_once()

    def test_splits_long_text(self, rag, tmp_path):
        rag.settings.vector_store_path = str(tmp_path / "vs")
        rag.settings.chunk_size = 50
        rag.settings.chunk_overlap = 0
        long_text = "word " * 100  # 500 chars
        rag.add_documents([long_text])
        docs = rag.vector_store.add_documents.call_args[0][0]
        assert len(docs) > 1  # Should be chunked


class TestChat:
    def test_demo_mode_yields_sse(self, rag):
        rag.vector_store.index.ntotal = 1
        rag.vector_store.similarity_search.return_value = [
            Document(page_content="Doc content"),
        ]
        rag.settings.demo_mode = True
        events = list(rag.chat("hello"))
        assert len(events) == 3
        token_event = json.loads(events[0].removeprefix("data: ").strip())
        assert token_event["type"] == "token"
        sources_event = json.loads(events[1].removeprefix("data: ").strip())
        assert sources_event["type"] == "sources"
        assert events[2].strip() == "data: [DONE]"

    def test_demo_mode_no_docs(self, rag):
        rag.settings.demo_mode = True
        rag.vector_store.index.ntotal = 0
        events = list(rag.chat("hello"))
        token_event = json.loads(events[0].removeprefix("data: ").strip())
        assert "Désolé" in token_event["content"]
