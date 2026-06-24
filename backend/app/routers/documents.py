"""
Document management endpoints: upload, list, retrieve, delete.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.utils.file_utils import validate_pdf_upload, save_upload
from app.services.pdf_processor import extract_pages, get_document_stats, PDFProcessingError
from app.services import document_store, vector_store
from app.services.chunking import chunk_document
from app.models.schemas import UploadResponse, DocumentListResponse, DocumentInfo, SearchRequest, SearchResponse, RetrievedChunk

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    file_bytes = await file.read()
    validate_pdf_upload(file, file_bytes)

    document_id, dest_path = save_upload(file_bytes, file.filename)

    try:
        pages = extract_pages(dest_path)
    except PDFProcessingError as e:
        # Clean up the saved file if we can't process it
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))

    document_store.register_document(document_id, file.filename, pages)
    stats = get_document_stats(pages)

    # Chunk + embed + index into the vector store
    chunks = chunk_document(pages, document_id)
    try:
        num_chunks = vector_store.index_chunks(chunks, document_id, file.filename)
    except RuntimeError as e:
        # Most likely a missing OPENAI_API_KEY - extraction still succeeded, surface clearly.
        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            num_pages=stats["num_pages"],
            total_characters=stats["total_characters"],
            num_chunks=0,
            status="extracted_not_indexed",
            message=f"Text extracted but indexing failed: {e}",
        )

    document_store.mark_indexed(document_id, num_chunks)

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        num_pages=stats["num_pages"],
        total_characters=stats["total_characters"],
        num_chunks=num_chunks,
        status="indexed",
        message=f"Document processed and indexed into the vector store ({num_chunks} chunks).",
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    docs = document_store.list_documents()
    return DocumentListResponse(documents=docs, count=len(docs))


@router.get("/{document_id}")
async def get_document(document_id: str):
    doc = document_store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    pages = document_store.load_pages(document_id)
    return {
        **doc,
        "pages_preview": [
            {"page_number": p.page_number, "char_count": p.char_count, "preview": p.text[:200]}
            for p in pages
        ],
    }


@router.post("/search", response_model=SearchResponse)
async def search_documents(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    results = vector_store.retrieve(
        query=req.query,
        top_k=req.top_k,
        document_ids=req.document_ids,
    )
    return SearchResponse(
        query=req.query,
        results=[RetrievedChunk(**r) for r in results],
        count=len(results),
    )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    deleted = document_store.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"status": "deleted", "document_id": document_id}
