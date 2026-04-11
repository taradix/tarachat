"""Unit tests for RAGSystem pure methods (no ML models required)."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from tarachat.config import Settings
from tarachat.models import ChatMessage
from tarachat.rag import RAGSystem, _split_by_pages


@pytest.fixture
def rag(tmp_path):
    """RAGSystem with mock ML components."""
    settings = Settings(vector_store_path=str(tmp_path / "vs"))
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = lambda msgs, **kw: "\n".join(m["content"] for m in msgs)
    return RAGSystem(
        settings=settings,
        device="cpu",
        embeddings=MagicMock(),
        vector_store=MagicMock(),
        tokenizer=tokenizer,
        model=MagicMock(),
        text_splitter=MagicMock(**{"split_text.side_effect": lambda text: [text]}),
    )


class TestSplitByPages:
    def test_single_page(self):
        text = "[Page 1]\nHello world"
        assert _split_by_pages(text) == [(1, "Hello world")]

    def test_multiple_pages(self):
        text = "[Page 1]\nFirst\n\n[Page 2]\nSecond"
        result = _split_by_pages(text)
        assert result == [(1, "First"), (2, "Second")]

    def test_no_markers(self):
        text = "Plain text without markers"
        assert _split_by_pages(text) == [(1, "Plain text without markers")]

    def test_empty_pages_skipped(self):
        text = "[Page 1]\nContent\n\n[Page 2]\n\n[Page 3]\nMore"
        result = _split_by_pages(text)
        assert len(result) == 2
        assert result[0] == (1, "Content")
        assert result[1] == (3, "More")


class TestBuildPrompt:
    def test_basic_prompt(self, rag):
        docs = [Document(page_content="Some context")]
        result = rag._build_prompt("What is X?", docs)
        assert "Some context" in result
        assert "Question : What is X?" in result

    def test_prompt_without_history(self, rag):
        docs = [Document(page_content="ctx")]
        rag._build_prompt("Q?", docs)
        messages = rag.tokenizer.apply_chat_template.call_args[0][0]
        assert len(messages) == 2  # system + user only
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_prompt_with_history(self, rag):
        docs = [Document(page_content="ctx")]
        history = [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
        ]
        rag._build_prompt("Q?", docs, history)
        messages = rag.tokenizer.apply_chat_template.call_args[0][0]
        assert any(m["role"] == "user" and m["content"] == "Hi" for m in messages)
        assert any(m["role"] == "assistant" and m["content"] == "Hello" for m in messages)

    def test_prompt_truncates_history_to_6(self, rag):
        docs = [Document(page_content="ctx")]
        history = [ChatMessage(role="user", content=f"msg{i}") for i in range(10)]
        rag._build_prompt("Q?", docs, history)
        messages = rag.tokenizer.apply_chat_template.call_args[0][0]
        contents = [m["content"] for m in messages]
        assert any("msg4" in c for c in contents)
        assert any("msg9" in c for c in contents)
        assert not any("msg3" in c for c in contents)

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
    def test_source_returns_structured_dict(self, rag):
        docs = [Document(page_content="[Page 3]\nHello world from document", metadata={"filename": "test.pdf"})]
        sources = rag._extract_sources(docs)
        assert len(sources) == 1
        assert sources[0]["filename"] == "test.pdf"
        assert sources[0]["page"] == 3
        assert any("Hello world" in h for h in sources[0]["highlights"])

    def test_multiple_sources(self, rag):
        docs = [
            Document(page_content="Doc A", metadata={"filename": "a.pdf"}),
            Document(page_content="Doc B", metadata={"filename": "b.pdf"}),
        ]
        sources = rag._extract_sources(docs)
        assert len(sources) == 2

    def test_highlight_truncated(self, rag):
        docs = [Document(page_content="x" * 200, metadata={"filename": "big.pdf"})]
        sources = rag._extract_sources(docs)
        assert len(sources[0]["highlights"][0]) == 120

    def test_defaults_to_page_1_when_no_marker(self, rag):
        docs = [Document(page_content="No page marker here", metadata={})]
        sources = rag._extract_sources(docs)
        assert sources[0]["page"] == 1

    def test_deduplicates_by_filename_and_page(self, rag):
        docs = [
            Document(page_content="[Page 2]\nChunk A content here", metadata={"filename": "doc.pdf"}),
            Document(page_content="[Page 2]\nChunk B content here", metadata={"filename": "doc.pdf"}),
        ]
        sources = rag._extract_sources(docs)
        assert len(sources) == 1
        assert len(sources[0]["highlights"]) == 2

    def test_different_pages_not_deduplicated(self, rag):
        docs = [
            Document(page_content="[Page 1]\nFirst page", metadata={"filename": "doc.pdf"}),
            Document(page_content="[Page 2]\nSecond page", metadata={"filename": "doc.pdf"}),
        ]
        sources = rag._extract_sources(docs)
        assert len(sources) == 2


class TestRetrieveDocuments:
    def test_empty_index_returns_empty(self, rag):
        rag.vector_store.index.ntotal = 0
        assert rag.retrieve_documents("query") == []

    def test_uses_settings_top_k(self, rag):
        rag.vector_store.index.ntotal = 5
        doc = Document(page_content="hit")
        rag.vector_store.similarity_search_with_score.return_value = [(doc, 0.5)]
        result = rag.retrieve_documents("query")
        rag.vector_store.similarity_search_with_score.assert_called_once_with("query", k=rag.settings.top_k)
        assert len(result) == 1

    def test_custom_k(self, rag):
        rag.vector_store.index.ntotal = 5
        rag.vector_store.similarity_search_with_score.return_value = []
        rag.retrieve_documents("query", k=7)
        rag.vector_store.similarity_search_with_score.assert_called_once_with("query", k=7)

    def test_filters_above_threshold(self, rag):
        rag.vector_store.index.ntotal = 5
        rag.settings.similarity_threshold = 0.8
        doc_close = Document(page_content="close")
        doc_far = Document(page_content="far")
        rag.vector_store.similarity_search_with_score.return_value = [
            (doc_close, 0.5),
            (doc_far, 1.2),
        ]
        result = rag.retrieve_documents("query")
        assert result == [doc_close]


class TestAddDocuments:
    def test_empty_texts_is_noop(self, rag):
        rag.add_documents([])
        # No error, no vector store interaction

    def test_adds_and_saves(self, rag, tmp_path):
        rag.settings.vector_store_path = str(tmp_path / "vs")
        rag.add_documents(["hello world"], [{"source": "test"}])
        rag.vector_store.add_documents.assert_called_once()
        docs = rag.vector_store.add_documents.call_args[0][0]
        assert any("hello world" in d.page_content for d in docs)
        rag.vector_store.save_local.assert_called_once()

    def test_splits_long_text(self, rag, tmp_path):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        rag.settings.vector_store_path = str(tmp_path / "vs")
        rag.text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        long_text = "word " * 100  # 500 chars
        rag.add_documents([long_text])
        docs = rag.vector_store.add_documents.call_args[0][0]
        assert len(docs) > 1  # Should be chunked

    def test_all_chunks_have_page_metadata(self, rag, tmp_path):
        """Every chunk should have page metadata set, with no [Page N] in content."""
        rag.settings.vector_store_path = str(tmp_path / "vs")
        text = "[Page 5]\n" + "word " * 100
        rag.text_splitter = MagicMock(**{"split_text.side_effect": lambda t: [t[:50], t[50:]]})
        rag.add_documents([text], [{"filename": "test.pdf"}])
        docs = rag.vector_store.add_documents.call_args[0][0]
        assert len(docs) > 1
        for doc in docs:
            assert doc.metadata["page"] == 5
            assert "[Page 5]" not in doc.page_content


class TestChat:
    def test_demo_mode_yields_events(self, rag):
        rag.vector_store.index.ntotal = 1
        rag.vector_store.similarity_search_with_score.return_value = [
            (Document(page_content="Doc content"), 0.5),
        ]
        rag.settings.demo_mode = True
        events = list(rag.chat("hello"))
        assert len(events) == 2
        assert events[0]["type"] == "token"
        assert events[1]["type"] == "sources"

    def test_demo_mode_no_docs(self, rag):
        rag.settings.demo_mode = True
        rag.vector_store.index.ntotal = 0
        events = list(rag.chat("hello"))
        assert "Désolé" in events[0]["content"]
