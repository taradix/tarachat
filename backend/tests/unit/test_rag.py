"""Unit tests for RAG pipeline components (no ML models required)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from tarachat.config import Settings
from tarachat.models import ChatMessage
from tarachat.rag import (
    LLMGenerator,
    PromptBuilder,
    RAGPipeline,
    Retriever,
    _extract_sources,
    _rrf_merge,
    _split_by_pages,
)


@pytest.fixture
def settings(tmp_path):
    return Settings(vector_store_path=str(tmp_path / "vs"))


@pytest.fixture
def retriever(settings):
    return Retriever(settings=settings, vector_store=MagicMock(), bm25_retriever=None)


@pytest.fixture
def prompt_builder(settings):
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = lambda msgs, **kw: "\n".join(m["content"] for m in msgs)
    return PromptBuilder(settings=settings, tokenizer=tokenizer)


@pytest.fixture
def llm_generator(settings):
    return LLMGenerator(settings=settings, tokenizer=MagicMock(), model=MagicMock(), device="cpu")


@pytest.fixture
def pipeline(settings, retriever, prompt_builder, llm_generator):
    return RAGPipeline(
        settings=settings,
        text_splitter=MagicMock(**{"split_text.side_effect": lambda text: [text]}),
        embeddings=MagicMock(),
        retriever=retriever,
        prompt_builder=prompt_builder,
        generator=llm_generator,
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


class TestRRFMerge:
    def test_doc_in_both_lists_ranks_first(self):
        shared = Document(page_content="shared")
        only_a = Document(page_content="only_a")
        only_b = Document(page_content="only_b")
        result = _rrf_merge([[shared, only_a], [shared, only_b]], top_k=3, weights=[0.5, 0.5])
        assert result[0].page_content == "shared"

    def test_top_k_limits_results(self):
        docs = [Document(page_content=str(i)) for i in range(5)]
        result = _rrf_merge([docs], top_k=2, weights=[1.0])
        assert len(result) == 2

    def test_weight_influences_ranking(self):
        bm25_doc = Document(page_content="bm25")
        dense_doc = Document(page_content="dense")
        result = _rrf_merge([[bm25_doc], [dense_doc]], top_k=2, weights=[0.9, 0.1])
        assert result[0].page_content == "bm25"

    def test_empty_lists_returns_empty(self):
        assert _rrf_merge([[], []], top_k=5, weights=[0.5, 0.5]) == []


class TestExtractSources:
    def test_source_returns_structured_dict(self):
        docs = [Document(page_content="[Page 3]\nHello world from document", metadata={"filename": "test.pdf"})]
        sources = _extract_sources(docs)
        assert len(sources) == 1
        assert sources[0]["filename"] == "test.pdf"
        assert sources[0]["page"] == 3
        assert any("Hello world" in h for h in sources[0]["highlights"])

    def test_multiple_sources(self):
        docs = [
            Document(page_content="Doc A", metadata={"filename": "a.pdf"}),
            Document(page_content="Doc B", metadata={"filename": "b.pdf"}),
        ]
        sources = _extract_sources(docs)
        assert len(sources) == 2

    def test_highlight_truncated(self):
        docs = [Document(page_content="x" * 200, metadata={"filename": "big.pdf"})]
        sources = _extract_sources(docs)
        assert len(sources[0]["highlights"][0]) == 120

    def test_defaults_to_page_1_when_no_marker(self):
        docs = [Document(page_content="No page marker here", metadata={})]
        sources = _extract_sources(docs)
        assert sources[0]["page"] == 1

    def test_deduplicates_by_filename_and_page(self):
        docs = [
            Document(page_content="[Page 2]\nChunk A content here", metadata={"filename": "doc.pdf"}),
            Document(page_content="[Page 2]\nChunk B content here", metadata={"filename": "doc.pdf"}),
        ]
        sources = _extract_sources(docs)
        assert len(sources) == 1
        assert len(sources[0]["highlights"]) == 2

    def test_different_pages_not_deduplicated(self):
        docs = [
            Document(page_content="[Page 1]\nFirst page", metadata={"filename": "doc.pdf"}),
            Document(page_content="[Page 2]\nSecond page", metadata={"filename": "doc.pdf"}),
        ]
        sources = _extract_sources(docs)
        assert len(sources) == 2


class TestRetriever:
    def test_empty_index_returns_empty(self, retriever):
        retriever.vector_store.index.ntotal = 0
        assert retriever.retrieve("query") == []

    def test_uses_settings_top_k(self, retriever):
        retriever.vector_store.index.ntotal = 5
        doc = Document(page_content="hit")
        retriever.vector_store.similarity_search_with_score.return_value = [(doc, 0.5)]
        result = retriever.retrieve("query")
        retriever.vector_store.similarity_search_with_score.assert_called_once_with(
            "query", k=retriever.settings.top_k
        )
        assert len(result) == 1

    def test_custom_k(self, retriever):
        retriever.vector_store.index.ntotal = 5
        retriever.vector_store.similarity_search_with_score.return_value = []
        retriever.retrieve("query", k=7)
        retriever.vector_store.similarity_search_with_score.assert_called_once_with("query", k=7)

    def test_filters_above_threshold(self, retriever):
        retriever.vector_store.index.ntotal = 5
        retriever.settings.similarity_threshold = 0.8
        doc_close = Document(page_content="close")
        doc_far = Document(page_content="far")
        retriever.vector_store.similarity_search_with_score.return_value = [
            (doc_close, 0.5),
            (doc_far, 1.2),
        ]
        result = retriever.retrieve("query")
        assert result == [doc_close]

    def test_uses_hybrid_when_bm25_available(self, retriever):
        doc_bm25 = Document(page_content="keyword match", metadata={"page": 1})
        doc_dense = Document(page_content="semantic match", metadata={"page": 2})
        retriever.vector_store.index.ntotal = 2
        retriever.bm25_retriever = MagicMock(**{"invoke.return_value": [doc_bm25]})
        retriever.vector_store.similarity_search.return_value = [doc_dense]
        result = retriever.retrieve("query")
        retriever.bm25_retriever.invoke.assert_called_once_with("query")
        retriever.vector_store.similarity_search.assert_called_once()
        assert len(result) == 2

    def test_falls_back_to_dense_when_no_bm25(self, retriever):
        doc = Document(page_content="dense only")
        retriever.vector_store.index.ntotal = 1
        retriever.bm25_retriever = None
        retriever.vector_store.similarity_search_with_score.return_value = [(doc, 0.3)]
        result = retriever.retrieve("query")
        retriever.vector_store.similarity_search_with_score.assert_called_once()
        assert result == [doc]

    def test_bm25_k_updated_for_custom_k(self, retriever):
        retriever.vector_store.index.ntotal = 5
        retriever.bm25_retriever = MagicMock(**{"invoke.return_value": []})
        retriever.vector_store.similarity_search.return_value = []
        retriever.retrieve("query", k=7)
        assert retriever.bm25_retriever.k == 7

    def test_bm25_rebuilt_after_add_documents(self, retriever):
        doc = Document(page_content="test content", metadata={"page": 1})
        retriever.vector_store.docstore._dict = {"id1": doc}
        with patch("tarachat.rag.BM25Retriever") as mock_bm25_cls:
            mock_retriever = MagicMock()
            mock_bm25_cls.from_documents.return_value = mock_retriever
            retriever.add_documents([doc])
            mock_bm25_cls.from_documents.assert_called_once()
            assert retriever.bm25_retriever is mock_retriever

    def test_bm25_none_when_no_docs(self, retriever):
        retriever.vector_store.docstore._dict = {}
        retriever.add_documents([Document(page_content="some text", metadata={"page": 1})])
        assert retriever.bm25_retriever is None


class TestPromptBuilder:
    def test_basic_prompt(self, prompt_builder):
        docs = [Document(page_content="Some context")]
        result = prompt_builder.build("What is X?", docs)
        assert "Some context" in result
        assert "Question : What is X?" in result

    def test_prompt_without_history(self, prompt_builder):
        docs = [Document(page_content="ctx")]
        prompt_builder.build("Q?", docs)
        messages = prompt_builder.tokenizer.apply_chat_template.call_args[0][0]
        assert len(messages) == 2  # system + user only
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_prompt_with_history(self, prompt_builder):
        docs = [Document(page_content="ctx")]
        history = [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
        ]
        prompt_builder.build("Q?", docs, history)
        messages = prompt_builder.tokenizer.apply_chat_template.call_args[0][0]
        assert any(m["role"] == "user" and m["content"] == "Hi" for m in messages)
        assert any(m["role"] == "assistant" and m["content"] == "Hello" for m in messages)

    def test_prompt_truncates_history(self, prompt_builder):
        docs = [Document(page_content="ctx")]
        history = [ChatMessage(role="user", content=f"msg{i}") for i in range(10)]
        prompt_builder.build("Q?", docs, history)
        messages = prompt_builder.tokenizer.apply_chat_template.call_args[0][0]
        contents = [m["content"] for m in messages]
        assert any("msg4" in c for c in contents)
        assert any("msg9" in c for c in contents)
        assert not any("msg3" in c for c in contents)

    def test_multiple_docs_joined(self, prompt_builder):
        docs = [
            Document(page_content="First doc"),
            Document(page_content="Second doc"),
        ]
        result = prompt_builder.build("Q?", docs)
        assert "First doc" in result
        assert "Second doc" in result


class TestLLMGenerator:
    def test_demo_response_with_docs(self, llm_generator):
        docs = [Document(page_content="Important content here")]
        result = llm_generator.demo_response(docs)
        assert "Important content here" in result
        assert "trouvé" in result

    def test_demo_response_with_two_docs(self, llm_generator):
        docs = [
            Document(page_content="First"),
            Document(page_content="Second"),
        ]
        result = llm_generator.demo_response(docs)
        assert "First" in result
        assert "Second" in result

    def test_demo_response_no_docs(self, llm_generator):
        result = llm_generator.demo_response([])
        assert "Désolé" in result

    def test_demo_response_long_doc_truncated(self, llm_generator):
        docs = [Document(page_content="x" * 500)]
        result = llm_generator.demo_response(docs)
        assert len(result) < 500


class TestRAGPipeline:
    def test_empty_texts_is_noop(self, pipeline):
        pipeline.add_documents([])

    def test_adds_and_saves(self, pipeline):
        pipeline.add_documents(["hello world"], [{"source": "test"}])
        pipeline.retriever.vector_store.add_documents.assert_called_once()
        docs = pipeline.retriever.vector_store.add_documents.call_args[0][0]
        assert any("hello world" in d.page_content for d in docs)
        pipeline.retriever.vector_store.save_local.assert_called_once()

    def test_splits_long_text(self, pipeline):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        pipeline.text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        long_text = "word " * 100  # 500 chars
        pipeline.add_documents([long_text])
        docs = pipeline.retriever.vector_store.add_documents.call_args[0][0]
        assert len(docs) > 1

    def test_all_chunks_have_page_metadata(self, pipeline):
        text = "[Page 5]\n" + "word " * 100
        pipeline.text_splitter = MagicMock(**{"split_text.side_effect": lambda t: [t[:50], t[50:]]})
        pipeline.add_documents([text], [{"filename": "test.pdf"}])
        docs = pipeline.retriever.vector_store.add_documents.call_args[0][0]
        assert len(docs) > 1
        for doc in docs:
            assert doc.metadata["page"] == 5
            assert "[Page 5]" not in doc.page_content

    def test_demo_mode_yields_events(self, pipeline):
        pipeline.retriever.vector_store.index.ntotal = 1
        pipeline.retriever.vector_store.similarity_search_with_score.return_value = [
            (Document(page_content="Doc content"), 0.5),
        ]
        pipeline.settings.demo_mode = True
        events = list(pipeline.chat("hello"))
        assert len(events) == 2
        assert events[0]["type"] == "token"
        assert events[1]["type"] == "sources"

    def test_demo_mode_no_docs(self, pipeline):
        pipeline.settings.demo_mode = True
        pipeline.retriever.vector_store.index.ntotal = 0
        events = list(pipeline.chat("hello"))
        assert "Désolé" in events[0]["content"]
