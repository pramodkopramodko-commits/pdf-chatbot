"""
Pydantic models (request/response schemas) — updated for Phase 3.

Changes vs Phase 2:
  - Added: CreateSessionRequest, CreateSessionResponse, SessionInfo
  - Added: ChatRequest, ChatResponse, Source
  - DocumentInfo gets optional num_chunks field (was already stored in registry)
"""
from pydantic import BaseModel, Field
from typing import List, Optional


# ── Document schemas (unchanged from Phase 2) ─────────────────────────────────

class PageText(BaseModel):
    page_number: int
    text: str
    char_count: int


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    num_pages: int
    total_characters: int
    num_chunks: int = 0
    status: str
    message: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    num_pages: int
    uploaded_at: str
    indexed: bool = False
    num_chunks: int = 0


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    count: int


class ErrorResponse(BaseModel):
    detail: str


class RetrievedChunk(BaseModel):
    text: str
    document_id: str
    filename: str
    page_numbers: List[int]
    similarity: float
    keyword_score: int


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    document_ids: Optional[List[str]] = None


class SearchResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]
    count: int


# ── Chat schemas (Phase 3) ─────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Scope retrieval to these document IDs. Leave empty to search all documents.",
    )


class CreateSessionResponse(BaseModel):
    session_id: str
    document_ids: List[str]


class SessionInfo(BaseModel):
    session_id: str
    turn_count: int
    document_ids: List[str]


class Source(BaseModel):
    document_id: str
    filename: str
    page_numbers: List[int]
    excerpt: str
    similarity: float


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session ID returned by POST /api/chat/sessions")
    question: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str
    turn: int
