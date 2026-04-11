import io

import fitz
import pytest

from tarachat.pdf import _clean_text, extract_text, serve, validate


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


def _make_pdf_with_positions(items: list[tuple[float, str]], height: float = 720.0) -> bytes:
    """Build a PDF with text at specified y-positions on a tall page.

    Uses height=720 so a 5% margin spans 36 px — enough to place
    header/footer text clearly inside or outside the margin zone.
    """
    doc = fitz.open()
    page = doc.new_page(width=72, height=height)
    for y, text in items:
        page.insert_text((10, y), text, fontsize=8)
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


class TestCleanText:
    def test_rejoins_hyphenated_words(self):
        assert _clean_text("règle-\nment") == "règlement"

    def test_preserves_hyphen_not_at_line_break(self):
        assert _clean_text("bien-être") == "bien-être"

    def test_collapses_spaces(self):
        assert _clean_text("text   with   spaces") == "text with spaces"

    def test_collapses_blank_lines(self):
        assert _clean_text("line1\n\n\n\nline2") == "line1\n\nline2"

    def test_strips(self):
        assert _clean_text("  hello  ") == "hello"


class TestExtractText:
    def test_invalid_pdf_raises(self):
        with pytest.raises(ValueError, match="Failed to process PDF"):
            extract_text(b"not a pdf")

    def test_empty_bytes_raises(self):
        with pytest.raises(ValueError):
            extract_text(b"")

    def test_blank_pdf_raises_no_content(self):
        with pytest.raises(ValueError, match="No text content"):
            extract_text(_make_pdf())

    def test_pdf_with_text(self):
        text, metadata = extract_text(_make_pdf("Hello world"))
        assert "Hello world" in text
        assert "[Page 1]" in text
        assert metadata["num_pages"] == 1
        assert metadata["file_type"] == "pdf"

    def test_body_text_included(self):
        # y=360 is 50% down a 720-height page — well inside the body
        pdf = _make_pdf_with_positions([(360, "Article 1. Le règlement")])
        text, _ = extract_text(pdf)
        assert "Article 1" in text

    def test_header_excluded(self):
        # y=14 puts the block centre at ~2% — inside the top 5% margin
        pdf = _make_pdf_with_positions([(14, "Ville de X"), (360, "Corps du texte")])
        text, _ = extract_text(pdf)
        assert "Corps du texte" in text
        assert "Ville de X" not in text

    def test_footer_excluded(self):
        # y=706 puts the block centre at ~98% — inside the bottom 5% margin
        pdf = _make_pdf_with_positions([(706, "Page 1 / 42"), (360, "Corps du texte")])
        text, _ = extract_text(pdf)
        assert "Corps du texte" in text
        assert "Page 1 / 42" not in text

    def test_custom_margin(self):
        # With margin=0.40, text at y=200 (28% down) is in the top margin
        pdf = _make_pdf_with_positions([(200, "Filtered"), (360, "Kept")])
        text, _ = extract_text(pdf, margin=0.40)
        assert "Kept" in text
        assert "Filtered" not in text


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
