import io
import logging
from typing import Optional
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDF files and extract text content."""

    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, dict]:
        """
        Extract text from PDF bytes.

        Args:
            pdf_bytes: PDF file content as bytes

        Returns:
            Tuple of (extracted_text, metadata)

        Raises:
            ValueError: If PDF cannot be read or is empty
        """
        try:
            # Create a PDF reader from bytes
            pdf_file = io.BytesIO(pdf_bytes)
            reader = PdfReader(pdf_file)

            # Extract metadata
            metadata = {
                "num_pages": len(reader.pages),
                "file_type": "pdf"
            }

            # Try to get PDF metadata
            if reader.metadata:
                if reader.metadata.title:
                    metadata["title"] = reader.metadata.title
                if reader.metadata.author:
                    metadata["author"] = reader.metadata.author
                if reader.metadata.subject:
                    metadata["subject"] = reader.metadata.subject
                if reader.metadata.creator:
                    metadata["creator"] = reader.metadata.creator

            # Extract text from all pages
            text_content = []
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_content.append(f"[Page {page_num}]\n{page_text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue

            if not text_content:
                raise ValueError("No text content could be extracted from the PDF")

            full_text = "\n\n".join(text_content)

            logger.info(
                f"Successfully extracted {len(full_text)} characters "
                f"from {metadata['num_pages']} pages"
            )

            return full_text, metadata

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise ValueError(f"Failed to process PDF: {str(e)}")

    @staticmethod
    def validate_pdf(pdf_bytes: bytes) -> bool:
        """
        Validate if the file is a valid PDF.

        Args:
            pdf_bytes: File content as bytes

        Returns:
            True if valid PDF, False otherwise
        """
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            reader = PdfReader(pdf_file)
            # Try to access pages to ensure it's valid
            _ = len(reader.pages)
            return True
        except Exception as e:
            logger.warning(f"PDF validation failed: {e}")
            return False


# Create a global instance
pdf_processor = PDFProcessor()
