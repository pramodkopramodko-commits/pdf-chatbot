"""
Lightweight document registry.

Phase 1 only extracts and stores text — there's no vector DB yet (that's Phase 2).
We persist extracted per-page text + metadata as JSON files next to the uploaded PDF,
so Phase 2 can pick them up directly for chunking/embedding without re-parsing PDFs.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from app.config import settings
from app.models.schemas import PageText, DocumentInfo

REGISTRY_FILE = settings.UPLOAD_DIR / "_registry.json"


def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {}


def _save_registry(data: dict) -> None:
    REGISTRY_FILE.write_text(json.dumps(data, indent=2))


def register_document(document_id: str, filename: str, pages: List[PageText]) -> None:
    registry = _load_registry()
    registry[document_id] = {
        "document_id": document_id,
        "filename": filename,
        "num_pages": len(pages),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "indexed": False,  # will flip to True in Phase 2 after embedding
    }
    _save_registry(registry)

    # Persist extracted page text alongside the PDF for Phase 2 consumption
    pages_path = settings.UPLOAD_DIR / f"{document_id}__pages.json"
    pages_path.write_text(json.dumps([p.model_dump() for p in pages], indent=2))


def list_documents() -> List[DocumentInfo]:
    registry = _load_registry()
    return [DocumentInfo(**v) for v in registry.values()]


def get_document(document_id: str) -> Optional[dict]:
    registry = _load_registry()
    return registry.get(document_id)


def mark_indexed(document_id: str, num_chunks: int) -> None:
    registry = _load_registry()
    if document_id in registry:
        registry[document_id]["indexed"] = True
        registry[document_id]["num_chunks"] = num_chunks
        _save_registry(registry)


def load_pages(document_id: str) -> List[PageText]:
    pages_path = settings.UPLOAD_DIR / f"{document_id}__pages.json"
    if not pages_path.exists():
        return []
    raw = json.loads(pages_path.read_text())
    return [PageText(**p) for p in raw]


def delete_document(document_id: str) -> bool:
    registry = _load_registry()
    if document_id not in registry:
        return False
    del registry[document_id]
    _save_registry(registry)

    # Remove files
    for p in settings.UPLOAD_DIR.glob(f"{document_id}__*"):
        p.unlink(missing_ok=True)

    # Remove vector store chunks (import here to avoid circular import at module load time)
    from app.services import vector_store
    vector_store.delete_document_chunks(document_id)

    return True
