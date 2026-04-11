"""Unit tests for config module."""

from tarachat.config import get_settings


class TestGetSettings:
    def test_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
