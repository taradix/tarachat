"""Unit tests for config module."""

import pytest

from tarachat.config import Settings, get_settings

_SETTINGS_KEYS = [
    "HOST", "PORT", "CORS_ORIGINS",
    "MODEL_NAME", "EMBEDDING_MODEL",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "SIMILARITY_THRESHOLD",
    "DEMO_MODE", "MAX_TOKENS",
    "DATA_PATH", "VECTOR_STORE_PATH",
]


@pytest.fixture()
def isolated_settings(monkeypatch):
    """Return a Settings instance unaffected by the local environment or .env."""
    for key in _SETTINGS_KEYS:
        monkeypatch.delenv(key, raising=False)
    return Settings(_env_file=None)


class TestSettings:
    def test_defaults(self, isolated_settings):
        s = isolated_settings
        assert s.host == "0.0.0.0"
        assert s.port == 8000
        assert s.demo_mode is False
        assert s.chunk_size == 512
        assert s.chunk_overlap == 50
        assert s.top_k == 3
        assert s.similarity_threshold is None
        assert s.max_tokens == 128

    def test_cors_origins_default(self, isolated_settings):
        assert isolated_settings.cors_origins == "http://localhost:5173"


class TestGetSettings:
    def test_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
