"""
Chat endpoints — Phase 3.

Routes
------
POST /api/chat/sessions               Create a new session (optionally scoped to document_ids)
GET  /api/chat/sessions               List all active sessions
GET  /api/chat/sessions/{session_id}  Get session metadata + message history
DELETE /api/chat/sessions/{session_id} Delete a session

POST /api/chat                        Non-streaming chat (complete answer returned at once)
POST /api/chat/stream                 Streaming chat (Server-Sent Events)
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.services import chat as chat_service
from app.models.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    ChatRequest,
    ChatResponse,
    Source,
    SessionInfo,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])


# ── Session management ────────────────────────────────────────────────────────

@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(req: CreateSessionRequest):
    session_id = chat_service.create_session(document_ids=req.document_ids)
    return CreateSessionResponse(session_id=session_id, document_ids=req.document_ids or [])


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    return chat_service.list_sessions()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = chat_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id": session_id,
        "document_ids": session["document_ids"],
        "turn_count": len(session["messages"]) // 2,
        "messages": session["messages"],
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    deleted = chat_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "deleted", "session_id": session_id}


# ── Chat ──────────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Non-streaming chat.  Returns the full answer + structured sources in one response.
    Useful for simple integrations that don't support SSE / streaming.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = chat_service.answer_question(
            session_id=req.session_id,
            question=req.question,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        session_id=result["session_id"],
        turn=result["turn"],
    )


@router.post("/stream")
async def stream_chat(req: ChatRequest):
    """
    Streaming chat via Server-Sent Events (SSE).

    The response is a text/event-stream of JSON events:
      { type: "sources", payload: [...] }   — emitted first
      { type: "token",   payload: "text" }  — one per LLM token
      { type: "done",    payload: { turn } } — final event
      { type: "error",   payload: "msg" }   — on failure

    The client should accumulate "token" payloads to form the full answer.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session = chat_service.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    def event_generator():
        try:
            for chunk in chat_service.stream_answer(
                session_id=req.session_id,
                question=req.question,
            ):
                yield chunk
        except Exception as e:
            import json
            yield f"data: {json.dumps({'type': 'error', 'payload': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering for SSE
        },
    )
