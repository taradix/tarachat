"""Integration tests for RAGPipeline requiring real model downloads."""

from pathlib import Path

from tarachat.config import Settings
from tarachat.rag import RAGPipeline


def test_create_initializes_all_components(tmp_path):
    """RAGPipeline.create loads embeddings, vector store, tokenizer, and model."""
    settings = Settings(vector_store_path=str(tmp_path / "vs"))
    pipeline = RAGPipeline.create(settings=settings, device="cpu")

    assert pipeline.embeddings is not None
    assert pipeline.retriever.vector_store is not None
    assert pipeline.prompt_builder.tokenizer is not None
    assert pipeline.generator.model is not None


def test_create_persists_vector_store(tmp_path):
    """RAGPipeline.create saves the FAISS index to disk."""
    vs_path = tmp_path / "vs"
    settings = Settings(vector_store_path=str(vs_path))
    RAGPipeline.create(settings=settings, device="cpu")

    assert (vs_path / "index.faiss").exists()


def test_create_loads_existing_vector_store(tmp_path):
    """RAGPipeline.create reuses a previously saved vector store."""
    vs_path = tmp_path / "vs"
    settings = Settings(vector_store_path=str(vs_path))

    RAGPipeline.create(settings=settings, device="cpu")

    pipeline2 = RAGPipeline.create(settings=settings, device="cpu")
    assert pipeline2.retriever.vector_store is not None
    assert Path(vs_path / "index.faiss").exists()
