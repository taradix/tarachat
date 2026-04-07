import pytest
from tarachat.pdf_processor import PDFProcessor


@pytest.fixture
def processor():
    return PDFProcessor()


class TestValidatePdf:
    def test_invalid_bytes(self, processor):
        assert processor.validate_pdf(b"not a pdf") is False

    def test_empty_bytes(self, processor):
        assert processor.validate_pdf(b"") is False


class TestExtractTextFromPdf:
    def test_invalid_pdf_raises(self, processor):
        with pytest.raises(ValueError, match="Failed to process PDF"):
            processor.extract_text_from_pdf(b"not a pdf")

    def test_empty_bytes_raises(self, processor):
        with pytest.raises(ValueError):
            processor.extract_text_from_pdf(b"")
