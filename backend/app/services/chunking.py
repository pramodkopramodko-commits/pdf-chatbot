"""
Chunking service.

Strategy
--------
We chunk *within* the running text of a document, but every chunk keeps track of
exactly which page(s) it was drawn from. This matters because:
  - Naive whole-document chunking loses page boundaries, breaking source attribution.
  - Naive per-page-only chunking (no overlap across pages) can cut sentences awkwardly
    right at page breaks, hurting retrieval quality for ideas that span a page boundary.

Approach:
  1. Concatenate all pages' text into one string, tracking the (start_offset, end_offset)
     range owned by each page.
  2. Slide a window of CHUNK_SIZE characters with CHUNK_OVERLAP overlap across the full text.
  3. For each chunk, find which page(s) its character range overlaps and store all of them
     (usually one, occasionally two when a chunk straddles a page break).
  4. Trim chunks to break on whitespace where possible (avoid splitting mid-word) without
     significantly changing chunk size.
"""
from dataclasses import dataclass
from typing import List
from app.models.schemas import PageText
from app.config import settings


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_numbers: List[int]
    char_start: int
    char_end: int


def _build_full_text_with_offsets(pages: List[PageText]):
    """Concatenate page texts (separated by a newline) and record each page's char range."""
    full_text_parts = []
    offsets = []  # list of (page_number, start, end) in the concatenated string
    cursor = 0
    for page in pages:
        text = page.text
        start = cursor
        full_text_parts.append(text)
        cursor += len(text)
        end = cursor
        offsets.append((page.page_number, start, end))
        # separator between pages
        full_text_parts.append("\n")
        cursor += 1
    return "".join(full_text_parts), offsets


def _pages_for_range(offsets, start: int, end: int) -> List[int]:
    pages = []
    for page_number, p_start, p_end in offsets:
        if p_end <= start or p_start >= end:
            continue  # no overlap
        pages.append(page_number)
    return pages or [offsets[-1][0]]  # fallback: last page, should not normally happen


def _snap_to_whitespace(text: str, pos: int, search_window: int = 50) -> int:
    """Try to move `pos` to the nearest whitespace within `search_window` chars, to avoid
    cutting a chunk mid-word. Searches forward first, then backward."""
    if pos >= len(text):
        return len(text)
    for i in range(pos, min(pos + search_window, len(text))):
        if text[i].isspace():
            return i
    for i in range(pos, max(pos - search_window, 0), -1):
        if text[i].isspace():
            return i
    return pos


def chunk_document(pages: List[PageText], document_id: str) -> List[Chunk]:
    """Split a document's pages into overlapping chunks with page-number metadata."""
    full_text, offsets = _build_full_text_with_offsets(pages)
    chunk_size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP

    if len(full_text.strip()) == 0:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    text_len = len(full_text)

    while start < text_len:
        raw_end = min(start + chunk_size, text_len)
        end = _snap_to_whitespace(full_text, raw_end) if raw_end < text_len else raw_end

        chunk_text = full_text[start:end].strip()
        if chunk_text:
            page_numbers = _pages_for_range(offsets, start, end)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}_chunk_{idx}",
                    text=chunk_text,
                    page_numbers=page_numbers,
                    char_start=start,
                    char_end=end,
                )
            )
            idx += 1

        if end >= text_len:
            break

        # advance with overlap; guard against zero/negative progress
        next_start = end - overlap
        start = next_start if next_start > start else end

    return chunks
