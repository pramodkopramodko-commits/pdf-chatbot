# ── Stage 1: build deps ───────────────────────────────────────────────────────
# Use a specific slim image so the layer is reproducible.
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed only to compile some Python wheels (e.g. PyMuPDF, hnswlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home appuser
WORKDIR /home/appuser/app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Persistent storage directories (will be mounted as volumes in production)
RUN mkdir -p backend/storage/uploads backend/storage/chroma \
 && chown -R appuser:appuser .

USER appuser

# Render / Railway inject PORT; default to 8000 for local Docker runs.
ENV PORT=8000

# Uvicorn: bind to 0.0.0.0 so the container port is reachable; workers=1 keeps
# ChromaDB's in-process client thread-safe (a single process owns the DB file).
CMD ["sh", "-c", \
     "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers"]

# Tell Docker which port the app listens on (documentation only — Render reads $PORT)
EXPOSE 8000

# Health-check so orchestrators know when the container is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/api/health')" \
  || exit 1
