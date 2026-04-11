"""Unit tests for the CLI module."""

import json
from unittest.mock import MagicMock

import pytest

from tarachat.cli import _ask
from tarachat.models import ChatMessage


def _sse(*events):
    """Build SSE event strings from dicts (or raw strings for DONE)."""
    result = []
    for e in events:
        if isinstance(e, str):
            result.append(f"data: {e}\n\n")
        else:
            result.append(f"data: {json.dumps(e)}\n\n")
    return result


@pytest.fixture
def rag():
    return MagicMock()


class TestAsk:
    def test_returns_answer(self, rag, capsys):
        rag.chat.return_value = _sse(
            {"type": "token", "content": "Bonjour"},
            {"type": "token", "content": " monde"},
            {"type": "sources", "sources": []},
            "[DONE]",
        )
        result = _ask(rag, "hello", [])
        assert result == "Bonjour monde"

    def test_prints_tokens(self, rag, capsys):
        rag.chat.return_value = _sse(
            {"type": "token", "content": "Réponse"},
            {"type": "sources", "sources": []},
            "[DONE]",
        )
        _ask(rag, "hello", [])
        captured = capsys.readouterr()
        assert "Réponse" in captured.out

    def test_prints_sources(self, rag, capsys):
        rag.chat.return_value = _sse(
            {"type": "token", "content": "text"},
            {"type": "sources", "sources": [{"filename": "doc.pdf", "page": 3}]},
            "[DONE]",
        )
        _ask(rag, "hello", [])
        captured = capsys.readouterr()
        assert "doc.pdf#page=3" in captured.out

    def test_skips_sources_when_empty(self, rag, capsys):
        rag.chat.return_value = _sse(
            {"type": "token", "content": "text"},
            {"type": "sources", "sources": []},
            "[DONE]",
        )
        _ask(rag, "hello", [])
        captured = capsys.readouterr()
        assert "Sources:" not in captured.out

    def test_passes_history_to_rag(self, rag):
        rag.chat.return_value = _sse("[DONE]")
        history = [ChatMessage(role="user", content="before")]
        _ask(rag, "query", history)
        rag.chat.assert_called_once_with("query", history)

    def test_ignores_malformed_events(self, rag):
        rag.chat.return_value = [
            "data: not-json\n\n",
            *_sse({"type": "token", "content": "ok"}, "[DONE]"),
        ]
        result = _ask(rag, "hello", [])
        assert result == "ok"
