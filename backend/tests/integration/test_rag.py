"""Integration tests — real FAISS file I/O, no ML model loading."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from tarachat.config import Settings
from tarachat.rag import Retriever, _load_vector_store

_DIM = 8


@pytest.fixture
def embeddings():
    """Minimal embeddings stub accepted by LangChain's FAISS.

    LangChain's FAISS calls the embedding object as a callable — ``emb(texts)``
    — when it is not a recognised ``Embeddings`` instance.  Setting
    ``side_effect`` ensures the mock returns correctly shaped vectors.
    """
    m = MagicMock()
    m.embed_query.return_value = [0.1] * _DIM
    m.side_effect = lambda text: [0.1] * _DIM
    return m


class TestVectorStorePersistence:
    def test_creates_index_on_first_load(self, tmp_path, embeddings):
        """A missing vector store is created and written to disk."""
        path = tmp_path / "vs"
        _load_vector_store(path, embeddings)
        assert (path / "index.faiss").exists()
        assert (path / "index.pkl").exists()

    def test_warm_start_loads_existing_index(self, tmp_path, embeddings):
        """A second load reads the saved index rather than creating a new one."""
        path = tmp_path / "vs"
        store = _load_vector_store(path, embeddings)
        store.add_texts(["hello world"])
        store.save_local(str(path))

        store2 = _load_vector_store(path, embeddings)
        assert store2.index.ntotal == 1

    def test_documents_survive_round_trip(self, tmp_path, embeddings):
        """Documents added to a store are present after save + reload."""
        path = tmp_path / "vs"
        store = _load_vector_store(path, embeddings)
        store.add_texts(["persisted content"])
        store.save_local(str(path))

        reloaded = _load_vector_store(path, embeddings)
        assert reloaded.index.ntotal == 1


class TestRetrieverPersistence:
    def test_add_documents_saves_to_disk(self, tmp_path, embeddings):
        """Retriever.add_documents persists the updated index to disk."""
        settings = Settings(vector_store_path=str(tmp_path / "vs"))
        store = _load_vector_store(Path(settings.vector_store_path), embeddings)
        retriever = Retriever(settings=settings, vector_store=store)

        retriever.add_documents([Document(page_content="test", metadata={"page": 1})])

        assert (tmp_path / "vs" / "index.faiss").exists()
        assert store.index.ntotal == 1

    def test_add_documents_rebuilds_bm25(self, tmp_path, embeddings):
        """Retriever.add_documents creates a BM25 retriever from the new docs."""
        settings = Settings(vector_store_path=str(tmp_path / "vs"))
        store = _load_vector_store(Path(settings.vector_store_path), embeddings)
        retriever = Retriever(settings=settings, vector_store=store)
        assert retriever.bm25_retriever is None

        retriever.add_documents([Document(page_content="keyword match", metadata={"page": 1})])

        assert retriever.bm25_retriever is not None
