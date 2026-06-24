"""
Utility helpers for file validation, naming, and safe saving of uploads.
"""
import uuid
import re
from pathlib import Path
from fastapi import UploadFile, HTTPException
from app.config import settings


def validate_pdf_upload(file: UploadFile, file_bytes: bytes) -> None:
    """Validate extension, content-type and size of an uploaded PDF."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Only PDF files are supported. Got: {suffix or 'unknown'}")

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed is {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    # Basic magic-byte check for PDF
    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File does not appear to be a valid PDF.")


def sanitize_filename(filename: str) -> str:
    name = Path(filename).stem
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    return name[:80] if name else "document"


def generate_document_id() -> str:
    return uuid.uuid4().hex[:12]


def save_upload(file_bytes: bytes, original_filename: str) -> tuple[str, Path]:
    """Save the raw PDF bytes to disk under a generated document_id and return (document_id, path)."""
    document_id = generate_document_id()
    safe_name = sanitize_filename(original_filename)
    dest_path = settings.UPLOAD_DIR / f"{document_id}__{safe_name}.pdf"
    dest_path.write_bytes(file_bytes)
    return document_id, dest_path
