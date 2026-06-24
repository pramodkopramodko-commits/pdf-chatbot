"""
PDF processing service.

Responsible for extracting text from PDF files on a per-page basis.
Per-page granularity is preserved so that later phases (chunking + retrieval)
can attribute every chunk back to an exact page number for source citation.
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List
from app.models.schemas import PageText


class PDFProcessingError(Exception):
    pass


def extract_pages(pdf_path: Path) -> List[PageText]:
    """
    Extract text from every page of a PDF.
    Returns a list of PageText(page_number, text, char_count), 1-indexed pages.
    Raises PDFProcessingError if the file can't be opened or has no extractable pages.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFProcessingError(f"Could not open PDF: {e}")

    pages: List[PageText] = []
    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = text.strip()
            pages.append(PageText(page_number=i, text=text, char_count=len(text)))
    finally:
        doc.close()

    if len(pages) == 0:
        raise PDFProcessingError("PDF has no pages.")

    total_chars = sum(p.char_count for p in pages)
    if total_chars == 0:
        # Likely a scanned/image-only PDF with no extractable text layer.
        raise PDFProcessingError(
            "No extractable text found in this PDF. It may be a scanned document. "
            "OCR support is planned as a future enhancement."
        )

    return pages


def get_document_stats(pages: List[PageText]) -> dict:
    return {
        "num_pages": len(pages),
        "total_characters": sum(p.char_count for p in pages),
        "pages_with_text": sum(1 for p in pages if p.char_count > 0),
    }
