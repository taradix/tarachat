"""PDF manipulation for page selection and text highlighting."""

import io
import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def serve_pdf(
    pdf_path: Path,
    *,
    page: int | None = None,
    num_pages: int = 20,
    highlights: list[str] | None = None,
) -> bytes:
    """Read a PDF, optionally select pages around *page* and highlight text.

    Args:
        pdf_path: Path to the PDF file on disk.
        page: 1-based page number to centre the selection on.
        num_pages: How many pages to include around *page*.
        highlights: Text snippets to highlight with yellow annotations.

    Returns:
        Modified PDF content as bytes.
    """
    doc = fitz.open(str(pdf_path))

    if page is not None and doc.page_count > num_pages:
        min_page = max(page - num_pages // 2, 1)
        pages_before = page - min_page + 1
        max_page = min(page + num_pages - pages_before, doc.page_count)
        doc.select([p - 1 for p in range(min_page, max_page + 1)])
        logger.info("Selected pages %d-%d of %s", min_page, max_page, pdf_path.name)

    if highlights:
        for pdf_page in doc:
            for text in highlights:
                areas = pdf_page.search_for(text)
                if areas:
                    annot = pdf_page.add_highlight_annot(areas)
                    annot.update()

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()
