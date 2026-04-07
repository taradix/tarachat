import io
import logging

from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDF files and extract text content."""

    @staticmethod
    def _extract_metadata(reader: PdfReader) -> dict:
        """Extract metadata from a PDF reader."""
        metadata: dict = {
            "num_pages": len(reader.pages),
            "file_type": "pdf",
        }
        if reader.metadata:
            for key in ("title", "author", "subject", "creator"):
                value = getattr(reader.metadata, key, None)
                if value:
                    metadata[key] = value
        return metadata

    @staticmethod
    def _extract_pages(reader: PdfReader) -> list[str]:
        """Extract text from all pages."""
        text_content = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_content.append(f"[Page {page_num}]\n{page_text}")
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")
        return text_content

    @classmethod
    def extract_text_from_pdf(cls, pdf_bytes: bytes) -> tuple[str, dict]:
        """Extract text from PDF bytes.

        Args:
            pdf_bytes: PDF file content as bytes

        Returns:
            Tuple of (extracted_text, metadata)

        Raises:
            ValueError: If PDF cannot be read or is empty
        """
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
        except Exception as e:
            raise ValueError(f"Failed to process PDF: {e!s}") from e

        metadata = cls._extract_metadata(reader)
        text_content = cls._extract_pages(reader)

        if not text_content:
            raise ValueError("No text content could be extracted from the PDF")

        full_text = "\n\n".join(text_content)
        logger.info(
            f"Successfully extracted {len(full_text)} characters "
            f"from {metadata['num_pages']} pages"
        )
        return full_text, metadata

    @staticmethod
    def validate_pdf(pdf_bytes: bytes) -> bool:
        """Validate if the file is a valid PDF.

        Args:
            pdf_bytes: File content as bytes

        Returns:
            True if valid PDF, False otherwise
        """
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            _ = len(reader.pages)
        except Exception as e:
            logger.warning(f"PDF validation failed: {e}")
            return False
        else:
            return True


# Create a global instance
pdf_processor = PDFProcessor()
