"""
Central configuration — updated for Phase 4.

Changes:
  - UPLOAD_DIR and CHROMA_DIR can now be overridden via environment variables
    (needed for Render's persistent disk mounted at /data).
  - Added ALLOWED_ORIGINS for tighter CORS in production.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    # ── General ───────────────────────────────────────────────
    APP_NAME: str = "PDF Chatbot"
    ENV: str = os.getenv("ENV", "development")

    # ── File upload ───────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", 50))
    MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    ALLOWED_EXTENSIONS: set = {".pdf"}

    # Allow UPLOAD_DIR / CHROMA_DIR to be overridden via env (Render persistent disk)
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "storage" / "uploads")))
    CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "storage" / "chroma")))

    # ── Vector store ──────────────────────────────────────────
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "pdf_documents")

    # ── Chunking ──────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))

    # ── LLM / Embeddings ─────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    ANTHROPIC_CHAT_MODEL: str = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # ── Retrieval ─────────────────────────────────────────────
    TOP_K: int = int(os.getenv("TOP_K", 4))

    # ── CORS ─────────────────────────────────────────────────
    # In production set to your exact Render URL, e.g. "https://pdf-chatbot.onrender.com"
    # Leave as "*" for development / when you don't know the URL yet.
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    def ensure_dirs(self):
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
