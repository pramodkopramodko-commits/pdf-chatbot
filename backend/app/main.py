"""
FastAPI application entrypoint — updated for Phase 4.

Changes vs Phase 3:
  - CORS origins now respect settings.ALLOWED_ORIGINS (tighter in production).
  - Version bumped to 0.4.0.
  - Startup event logs key config so Render logs show the active provider/dirs.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.config import settings
from app.routers import documents, chat

logger = logging.getLogger("pdf_chatbot")
logging.basicConfig(level=logging.INFO)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

app = FastAPI(
    title=settings.APP_NAME,
    version="0.4.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


@app.on_event("startup")
async def startup_event():
    logger.info("=== PDF Chatbot starting up ===")
    logger.info(f"ENV          : {settings.ENV}")
    logger.info(f"LLM provider : {settings.LLM_PROVIDER}")
    logger.info(f"Chat model   : {settings.OPENAI_CHAT_MODEL if settings.LLM_PROVIDER == 'openai' else settings.ANTHROPIC_CHAT_MODEL}")
    logger.info(f"Upload dir   : {settings.UPLOAD_DIR}")
    logger.info(f"Chroma dir   : {settings.CHROMA_DIR}")
    logger.info(f"OPENAI key   : {'SET' if settings.OPENAI_API_KEY else 'MISSING'}")


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "templates" / "index.html"))


@app.get("/api/health")
async def health():
    from app.services.vector_store import collection_count
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": "0.4.0",
        "env": settings.ENV,
        "llm_provider": settings.LLM_PROVIDER,
        "vector_store_chunks": collection_count(),
    }
