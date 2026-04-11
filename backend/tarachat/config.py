from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    cors_origins: str = "http://localhost:5173"

    # Model Configuration
    model_name: str = "croissantllm/CroissantLLMChat-v0.1"
    embedding_model: str = "OrdalieTech/Solon-embeddings-large-0.1"

    # RAG Configuration
    chunk_size: int = 512  # tokens (capped at embedding model's max_seq_length)
    chunk_overlap: int = 50
    top_k: int = 5
    bm25_weight: float = 0.5  # BM25 share in hybrid retrieval; dense gets 1 - bm25_weight
    conversation_history_size: int = 6
    similarity_threshold: float | None = None  # Max L2 distance; None = no filtering (use logs to calibrate)

    # Performance Configuration
    demo_mode: bool = False  # Set to True for fast responses without LLM (testing/development)
    max_tokens: int = 128  # Reduced from 512 for faster generation

    # Paths
    data_path: str = "./data"
    vector_store_path: str = "./vector_store"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
