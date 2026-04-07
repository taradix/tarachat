import json
import sqlite3
from pathlib import Path

import pytest

from tarachat.ingest import DocumentManager


class FakeSettings:
    def __init__(self, tmp_path):
        self.vector_store_path = str(tmp_path / "vector_store")


class FakeVectorStore:
    def __init__(self):
        self.saved = False

    def save_local(self, path):
        self.saved = True


class FakeRAG:
    def __init__(self, tmp_path):
        self.settings = FakeSettings(tmp_path)
        self.vector_store = FakeVectorStore()
        self.added: list[tuple[list, list]] = []

    def add_documents(self, texts, metadatas=None):
        self.added.append((texts, metadatas))

    def create_empty_vector_store(self):
        return FakeVectorStore()


class FakePDF:
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> tuple[str, dict]:
        return "pdf content", {"pages": 1}


@pytest.fixture()
def rag(tmp_path):
    return FakeRAG(tmp_path)


@pytest.fixture()
def pdf():
    return FakePDF()


@pytest.fixture()
def manager(rag, pdf):
    return DocumentManager(rag, pdf)


class TestAddDocument:
    def test_add_stores_in_db_and_rag(self, manager, rag):
        result = manager.add_document("doc1", "hello world")
        assert result is True
        assert len(rag.added) == 1
        assert rag.added[0][0] == ["hello world"]

        with sqlite3.connect(manager.db_path) as conn:
            row = conn.execute("SELECT id, content FROM documents WHERE id = ?", ("doc1",)).fetchone()
        assert row is not None
        assert row[1] == "hello world"

    def test_add_duplicate_returns_false(self, manager):
        manager.add_document("doc1", "content")
        result = manager.add_document("doc1", "other")
        assert result is False

    def test_add_with_metadata(self, manager):
        manager.add_document("doc1", "content", {"author": "test"})
        with sqlite3.connect(manager.db_path) as conn:
            row = conn.execute("SELECT metadata FROM documents WHERE id = ?", ("doc1",)).fetchone()
        meta = json.loads(row[0])
        assert meta["author"] == "test"
        assert meta["doc_id"] == "doc1"


class TestDeleteDocument:
    def test_delete_existing(self, manager):
        manager.add_document("doc1", "content")
        result = manager.delete_document("doc1")
        assert result is True
        assert not manager._doc_exists("doc1")

    def test_delete_nonexistent_returns_false(self, manager):
        result = manager.delete_document("nope")
        assert result is False


class TestUpdateDocument:
    def test_update_existing(self, manager, rag):
        manager.add_document("doc1", "old")
        result = manager.update_document("doc1", "new")
        assert result is True

        with sqlite3.connect(manager.db_path) as conn:
            row = conn.execute("SELECT content FROM documents WHERE id = ?", ("doc1",)).fetchone()
        assert row[0] == "new"

    def test_update_nonexistent_returns_false(self, manager):
        result = manager.update_document("nope", "content")
        assert result is False

    def test_update_preserves_metadata(self, manager):
        manager.add_document("doc1", "old", {"key": "value"})
        manager.update_document("doc1", "new")

        with sqlite3.connect(manager.db_path) as conn:
            row = conn.execute("SELECT metadata FROM documents WHERE id = ?", ("doc1",)).fetchone()
        meta = json.loads(row[0])
        assert meta["key"] == "value"


class TestListDocuments:
    def test_list_empty(self, manager, capsys):
        manager.list_documents()

    def test_list_populated(self, manager):
        manager.add_document("a", "aaa")
        manager.add_document("b", "bbb")
        manager.list_documents()


class TestReadFileContent:
    def test_read_text_file(self, manager, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        content, meta = manager._read_file_content(f)
        assert content == "hello"
        assert meta == {}

    def test_read_pdf_file(self, manager, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"fake pdf")
        content, meta = manager._read_file_content(f)
        assert content == "pdf content"
        assert meta == {"pages": 1}


class TestAddFromDirectory:
    def test_adds_files(self, manager, rag, tmp_path):
        (tmp_path / "a.txt").write_text("doc a", encoding="utf-8")
        (tmp_path / "b.txt").write_text("doc b", encoding="utf-8")
        manager.add_from_directory(tmp_path, "*.txt")
        assert len(rag.added) == 2

    def test_skips_missing_dir(self, manager, rag, tmp_path):
        manager.add_from_directory(tmp_path / "nope", "*.txt")
        assert len(rag.added) == 0


class TestClearAll:
    def test_clear(self, manager, monkeypatch):
        manager.add_document("doc1", "content")
        monkeypatch.setattr("builtins.input", lambda _: "yes")
        manager.clear_all()
        assert not manager._doc_exists("doc1")

    def test_clear_cancelled(self, manager, monkeypatch):
        manager.add_document("doc1", "content")
        monkeypatch.setattr("builtins.input", lambda _: "no")
        manager.clear_all()
        assert manager._doc_exists("doc1")


class TestMigrateFromJson:
    def test_migrates_json_to_sqlite(self, rag, pdf, tmp_path):
        vs_path = Path(rag.settings.vector_store_path)
        vs_path.mkdir(parents=True)
        json_path = vs_path / "documents_metadata.json"
        json_path.write_text(json.dumps({
            "doc1": {"metadata": {"source": "test"}, "content_length": 42}
        }))

        manager = DocumentManager(rag, pdf)
        assert manager._doc_exists("doc1")
        assert not json_path.exists()
        assert json_path.with_suffix(".json.bak").exists()


class TestInitFromSampleFile:
    def test_loads_paragraphs(self, manager, rag, tmp_path):
        sample = tmp_path / "samples.txt"
        sample.write_text("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.", encoding="utf-8")
        manager.init_from_sample_file(sample)
        assert len(rag.added) == 1
        texts = rag.added[0][0]
        assert texts == ["First paragraph.", "Second paragraph.", "Third paragraph."]

    def test_missing_file(self, manager, rag, tmp_path):
        manager.init_from_sample_file(tmp_path / "nope.txt")
        assert len(rag.added) == 0

    def test_empty_file(self, manager, rag, tmp_path):
        sample = tmp_path / "empty.txt"
        sample.write_text("", encoding="utf-8")
        manager.init_from_sample_file(sample)
        assert len(rag.added) == 0
