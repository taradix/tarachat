from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Model Configuration
    model_name: str = "croissantllm/CroissantLLMChat-v0.1"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # RAG Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 3

    # Performance Configuration
    demo_mode: bool = True  # Set to True for fast responses without LLM (testing/development)
    max_tokens: int = 128  # Reduced from 512 for faster generation

    # Paths
    data_path: str = "./data"
    vector_store_path: str = "./vector_store"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
