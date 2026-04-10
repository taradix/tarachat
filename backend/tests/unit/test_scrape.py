"""Unit tests for scrape module (pure functions, no network)."""

import json
from pathlib import Path

import pytest
from yarl import URL

from tarachat.scrape import (
    Downloader,
    has_changed,
    load_metadata,
    meta_path_for,
    sanitize_filename,
    save_metadata,
)


class TestMetaPathFor:
    def test_appends_meta_json(self):
        result = meta_path_for(Path("/data/file.pdf"))
        assert result == Path("/data/file.pdf.meta.json")

    def test_txt_file(self):
        result = meta_path_for(Path("docs/readme.txt"))
        assert result == Path("docs/readme.txt.meta.json")


class TestLoadMetadata:
    def test_missing_file_returns_empty(self, tmp_path):
        result = load_metadata(tmp_path / "nonexistent.pdf")
        assert result == {}

    def test_reads_existing_metadata(self, tmp_path):
        file_path = tmp_path / "doc.pdf"
        meta_file = tmp_path / "doc.pdf.meta.json"
        meta_file.write_text(json.dumps({"etag": "abc"}), encoding="utf-8")
        result = load_metadata(file_path)
        assert result == {"etag": "abc"}

    def test_corrupt_json_returns_empty(self, tmp_path):
        file_path = tmp_path / "doc.pdf"
        meta_file = tmp_path / "doc.pdf.meta.json"
        meta_file.write_text("{bad json", encoding="utf-8")
        result = load_metadata(file_path)
        assert result == {}


class TestSaveMetadata:
    def test_writes_metadata(self, tmp_path):
        file_path = tmp_path / "doc.pdf"
        metadata = {"etag": '"xyz"', "last_modified": "Mon, 01 Jan 2024"}
        save_metadata(file_path, metadata)
        meta_file = tmp_path / "doc.pdf.meta.json"
        assert meta_file.exists()
        assert json.loads(meta_file.read_text(encoding="utf-8")) == metadata


class TestHasChanged:
    def test_empty_remote_meta_means_changed(self):
        assert has_changed({"etag": "a"}, {}) is True

    def test_different_etag(self):
        local = {"etag": "a", "last_modified": "x", "content_length": "100"}
        remote = {"etag": "b", "last_modified": "x", "content_length": "100"}
        assert has_changed(local, remote) is True

    def test_different_last_modified(self):
        local = {"etag": "a", "last_modified": "x", "content_length": "100"}
        remote = {"etag": "a", "last_modified": "y", "content_length": "100"}
        assert has_changed(local, remote) is True

    def test_different_content_length(self):
        local = {"etag": "a", "last_modified": "x", "content_length": "100"}
        remote = {"etag": "a", "last_modified": "x", "content_length": "200"}
        assert has_changed(local, remote) is True

    def test_same_metadata_no_change(self):
        meta = {"etag": "a", "last_modified": "x", "content_length": "100"}
        assert has_changed(meta, meta) is False


class FakeDownloader(Downloader):
    """A Downloader that performs no real I/O."""

    def __init__(self, *, remote_meta=None, fetch_error=None):
        self.remote_meta = remote_meta or {}
        self.fetch_error = fetch_error
        self.fetched: list[tuple] = []

    async def fetch_metadata(self, session, url):
        return self.remote_meta

    async def fetch_content(self, session, url, file_path, **_kwargs):
        if self.fetch_error:
            raise self.fetch_error
        self.fetched.append((url, file_path))
        file_path.write_bytes(b"fake content")


class TestDownloadOne:
    @pytest.mark.asyncio
    async def test_skips_unchanged(self, tmp_path):
        url = URL("http://example.com/file.txt")
        file_path = tmp_path / "file.txt"
        file_path.write_text("existing", encoding="utf-8")

        meta = {"etag": "abc", "last_modified": "Mon", "content_length": "8", "url": str(url)}
        save_metadata(file_path, meta)

        dl = FakeDownloader(remote_meta=meta)
        result = await dl.download_one(None, url, tmp_path)
        assert result[2] == "skipped"

    @pytest.mark.asyncio
    async def test_downloads_new_file(self, tmp_path):
        url = URL("http://example.com/newfile.txt")
        dl = FakeDownloader()
        result = await dl.download_one(None, url, tmp_path)
        assert result[2] == "downloaded"
        assert len(dl.fetched) == 1

    @pytest.mark.asyncio
    async def test_error_on_failed_download(self, tmp_path):
        url = URL("http://example.com/fail.txt")
        dl = FakeDownloader(fetch_error=Exception("timeout"))
        result = await dl.download_one(None, url, tmp_path)
        assert result[2] == "error"
        assert result[1] is None

    @pytest.mark.asyncio
    async def test_custom_filename_overrides_url_name(self, tmp_path):
        url = URL("http://example.com/Rglement-07-296.pdf")
        dl = FakeDownloader()
        result = await dl.download_one(
            None, url, tmp_path, filename="Règlement-07-296.pdf",
        )
        assert result[2] == "downloaded"
        assert result[1] == str(tmp_path / "Règlement-07-296.pdf")


class TestSanitizeFilename:
    def test_preserves_accented_characters(self):
        assert sanitize_filename("Règlement numéro 06", ".pdf") == "Règlement numéro 06.pdf"

    def test_replaces_forbidden_characters(self):
        assert sanitize_filename('a/b:c*d', '.txt') == 'a_b_c_d.txt'

    def test_does_not_duplicate_extension(self):
        assert sanitize_filename("readme.txt", ".txt") == "readme.txt"

    def test_empty_text_gets_placeholder(self):
        result = sanitize_filename("", ".pdf")
        assert result == "_.pdf"

    def test_no_extension(self):
        assert sanitize_filename("hello world") == "hello world"

    def test_fallback_when_only_dots(self):
        result = sanitize_filename("...", ".pdf")
        assert result == "_.pdf"

    def test_truncates_long_filename(self):
        long_text = "Règlement numéro 04.03.2013 " + "A" * 300
        result = sanitize_filename(long_text, ".pdf")
        # Must fit within 245 bytes (255 - len(".meta.json"))
        assert len(result.encode("utf-8")) <= 245
        assert result.endswith(".pdf")

    def test_truncated_filename_contains_hash(self):
        long_text = "A" * 300
        result = sanitize_filename(long_text, ".pdf")
        # The 8-char hex hash should appear before the extension
        import re
        assert re.search(r"_[0-9a-f]{8}\.pdf$", result)

    def test_different_long_texts_produce_different_names(self):
        a = sanitize_filename("A" * 300, ".pdf")
        b = sanitize_filename("B" * 300, ".pdf")
        assert a != b

    def test_short_filename_unchanged(self):
        name = "short.pdf"
        assert sanitize_filename("short", ".pdf") == name
