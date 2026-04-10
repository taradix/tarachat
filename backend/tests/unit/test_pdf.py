import io

import fitz
import pytest

from tarachat.pdf import extract_text, serve, validate


def _make_pdf(text: str = "") -> bytes:
    """Build a minimal in-memory PDF, optionally with text on a page."""
    doc = fitz.open()
    page = doc.new_page(width=72, height=72)
    if text:
        page.insert_text((10, 20), text, fontsize=8)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


class TestValidate:
    def test_invalid_bytes(self):
        assert validate(b"not a pdf") is False

    def test_empty_bytes(self):
        assert validate(b"") is False

    def test_valid_pdf(self):
        assert validate(_make_pdf()) is True


class TestExtractText:
    def test_invalid_pdf_raises(self):
        with pytest.raises(ValueError, match="Failed to process PDF"):
            extract_text(b"not a pdf")

    def test_empty_bytes_raises(self):
        with pytest.raises(ValueError):
            extract_text(b"")

    def test_blank_pdf_raises_no_content(self):
        """A valid PDF with a blank page has no extractable text."""
        with pytest.raises(ValueError, match="No text content"):
            extract_text(_make_pdf())

    def test_pdf_with_text(self):
        text, metadata = extract_text(_make_pdf("Hello world"))
        assert "Hello world" in text
        assert "[Page 1]" in text
        assert metadata["num_pages"] == 1
        assert metadata["file_type"] == "pdf"


class TestServe:
    def test_returns_pdf_bytes(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(_make_pdf("Hello"))
        result = serve(pdf_path)
        assert result[:5] == b"%PDF-"

    def test_page_selection(self, tmp_path):
        doc = fitz.open()
        for i in range(30):
            page = doc.new_page(width=72, height=72)
            page.insert_text((10, 20), f"Page {i + 1}", fontsize=8)
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        pdf_path = tmp_path / "multi.pdf"
        pdf_path.write_bytes(buf.getvalue())

        result = serve(pdf_path, page=15, num_pages=10)
        out_doc = fitz.open(stream=result, filetype="pdf")
        assert out_doc.page_count == 10
        out_doc.close()

    def test_highlights(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(_make_pdf("Important text"))
        result = serve(pdf_path, highlights=["Important"])
        assert len(result) > 0

    def test_no_page_selection_when_few_pages(self, tmp_path):
        pdf_path = tmp_path / "small.pdf"
        pdf_path.write_bytes(_make_pdf("Content"))
        result = serve(pdf_path, page=1, num_pages=20)
        out_doc = fitz.open(stream=result, filetype="pdf")
        assert out_doc.page_count == 1
        out_doc.close()
