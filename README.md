# PDF Chatbot

An AI-powered chatbot that lets you upload PDF documents and ask questions about their contents. Answers stream in real time and are grounded in your documents with exact page citations.

**Live demo:** `https://your-app.onrender.com` *(replace after deploying)*

---

## Project Status — All 4 Phases Complete ✅

| Phase | Scope | Status |
|---|---|---|
| 1 | FastAPI setup, PDF upload, per-page text extraction | ✅ Done |
| 2 | Chunking, OpenAI embeddings, ChromaDB vector store, hybrid retrieval | ✅ Done |
| 3 | Streaming chat (SSE), session memory, source attribution, full UI | ✅ Done |
| 4 | Docker, Render deployment, production config, final docs | ✅ Done |

---

## Features

- **PDF upload** — drag-and-drop, up to 50 MB, validated for type + size
- **Automatic indexing** — documents chunked, embedded, and stored in ChromaDB on upload
- **Streaming answers** — token-by-token SSE stream from GPT-4o-mini or Claude
- **Source attribution** — every answer links to the exact filename + page number(s)
- **Conversation memory** — follow-up questions work across a session (last 6 turns)
- **Multiple PDFs** — upload many documents; the chatbot searches across all of them
- **Hybrid retrieval** — vector similarity + keyword overlap re-ranking
- **Dual LLM support** — switch between OpenAI and Anthropic via one env var
- **Docker-ready** — single `docker compose up --build` for local dev
- **Render deployment** — `render.yaml` blueprint for one-click deploy

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser                                                        │
│  ┌──────────────┐  ┌─────────────────────────────────────────┐ │
│  │  Sidebar     │  │  Chat panel                             │ │
│  │  • Upload    │  │  • Streaming message bubbles            │ │
│  │  • Doc list  │  │  • Sources panel (📎 page citations)    │ │
│  │  • New Chat  │  │  • Input bar (Enter to send)            │ │
│  └──────────────┘  └─────────────────────────────────────────┘ │
└───────────────┬─────────────────────────┬───────────────────────┘
                │ REST / SSE              │
┌───────────────▼─────────────────────────▼───────────────────────┐
│  FastAPI (uvicorn)                                              │
│                                                                 │
│  POST /api/documents/upload                                     │
│    └─▶ file_utils        validate + save PDF                    │
│    └─▶ pdf_processor     PyMuPDF: per-page text extraction      │
│    └─▶ document_store    JSON registry + pages JSON             │
│    └─▶ chunking          overlapping page-aware chunks          │
│    └─▶ embeddings        OpenAI text-embedding-3-small (batch)  │
│    └─▶ vector_store      ChromaDB upsert (cosine index)         │
│                                                                 │
│  POST /api/chat/sessions  → create session, return session_id   │
│                                                                 │
│  POST /api/chat/stream  { session_id, question }               │
│    └─▶ vector_store.retrieve()   embed query → cosine search    │
│              └─▶ hybrid re-rank  similarity + keyword overlap   │
│    └─▶ chat._format_sources_block()  numbered <source> XML      │
│    └─▶ chat._build_*_messages()  system + history + sources     │
│    └─▶ OpenAI / Anthropic streaming API                         │
│    └─▶ SSE token stream  →  Browser accumulates into bubble     │
│                                                                 │
│  Persistent storage (volume / Render disk)                      │
│    backend/storage/uploads/   raw PDFs + pages JSON + registry  │
│    backend/storage/chroma/    ChromaDB SQLite + HNSW index      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design Decisions

### Chunking strategy
~1000-character sliding window with ~200-character overlap, snapped to whitespace boundaries. The chunker concatenates all page text into one string while tracking character offsets per page. This means chunks can span page boundaries (preventing mid-idea cuts) while still recording exactly which page(s) they came from — essential for source attribution.

### Embedding model
`text-embedding-3-small` (OpenAI). Best quality/cost ratio for general document retrieval. 1536-dimensional vectors; batched at 96 texts per API call. Embeddings always go through OpenAI even when `LLM_PROVIDER=anthropic`, because Anthropic has no embeddings API.

### Prompt design
Strict RAG pattern. The system prompt tells the LLM to answer **only** from the `<sources>` block injected into each user message and to cite pages inline. If the answer is absent from context, it must say so — not hallucinate. Retrieved chunks are numbered `<source index="N" file="..." page N>` blocks. The last 6 conversation turns are prepended for follow-up support.

### Retrieval approach
ChromaDB cosine similarity search over 1536-dim vectors. We over-fetch `TOP_K × 3` candidates, then re-rank by `similarity + 0.03 × keyword_overlap_count` and trim to `TOP_K`. The small keyword bonus catches exact-term matches (names, acronyms, codes) that pure vector search under-ranks — this is the "hybrid search" bonus feature.

### Streaming
`POST /api/chat/stream` returns `text/event-stream` (SSE). The service generator yields JSON events: `sources` (emitted before the first token so citations appear immediately), `token` (one per LLM chunk), `done`, and `error`. The frontend accumulates tokens into the bubble as they arrive.

### Session / conversation memory
In-process `dict` keyed by `session_id` UUID. Each session holds the raw `messages` list trimmed to the last 6 turns (12 messages). Sessions are scoped optionally to specific `document_ids` for multi-PDF workflows. In-process memory resets on server restart — for persistence across restarts a Redis or SQLite backend could be swapped in later.

### Storage
Two persistent directories: `uploads/` (raw PDFs + per-page JSON + JSON registry) and `chroma/` (ChromaDB SQLite + HNSW index). Both are mounted as named Docker volumes locally and as a Render persistent disk in production, so data survives container restarts and redeployments.

---

## Local Setup (without Docker)

### Prerequisites
- Python 3.10+
- An OpenAI API key (required for embeddings)

```bash
cd backend
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set OPENAI_API_KEY (and optionally ANTHROPIC_API_KEY + LLM_PROVIDER)

uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000**.

---

## Local Setup (Docker Compose) 🐳

```bash
cp backend/.env.example backend/.env
# Edit backend/.env — set OPENAI_API_KEY at minimum

docker compose up --build
```

Open **http://localhost:8000**.

Data is stored in Docker named volumes (`uploads` and `chroma`) — it survives `docker compose down` and comes back on `docker compose up`.

To fully reset storage:
```bash
docker compose down -v   # -v removes the named volumes
```

---

## Deployment on Render 🚀

### One-time setup

1. **Push to GitHub** — make the repo public (or grant Render access to a private repo).

2. **Create a Render account** at [render.com](https://render.com) if you don't have one.

3. **New → Blueprint** in the Render dashboard → connect your GitHub repo.
   Render auto-detects `render.yaml` and pre-fills the service config.

4. **Add secret environment variables** in the Render dashboard (these are marked `sync: false` in `render.yaml` so they're never stored in git):
   - `OPENAI_API_KEY` → your key
   - `ANTHROPIC_API_KEY` → your key (only if `LLM_PROVIDER=anthropic`)

5. Click **Apply** — Render builds the Docker image and deploys. First build takes ~3–5 minutes.

6. Your live URL will be `https://pdf-chatbot-XXXX.onrender.com`.
   Update `ALLOWED_ORIGINS` in the Render dashboard to this URL for tighter CORS.

### Persistent storage on Render
`render.yaml` provisions a **1 GB persistent disk** mounted at `/data`. The app writes uploads to `/data/uploads` and ChromaDB to `/data/chroma`. Without the disk, storage resets on every redeploy.

> **Free tier note:** Render's free tier does not support persistent disks. On free tier, uploaded PDFs and the vector index are lost on each redeploy — you'll need to re-upload. Upgrade to a paid plan ($7/mo) to get the disk.

### Updating after code changes
```bash
git add . && git commit -m "..." && git push
```
Render picks up the push and redeploys automatically (`autoDeploy: true` in `render.yaml`).

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/api/health` | Health check + stats |
| GET | `/api/docs` | Swagger UI (auto-generated) |
| POST | `/api/documents/upload` | Upload + index a PDF |
| GET | `/api/documents` | List all documents |
| GET | `/api/documents/{id}` | Document detail + page previews |
| DELETE | `/api/documents/{id}` | Delete document + remove from index |
| POST | `/api/documents/search` | Direct semantic search (retrieval test) |
| POST | `/api/chat/sessions` | Create a new chat session |
| GET | `/api/chat/sessions` | List active sessions |
| GET | `/api/chat/sessions/{id}` | Session detail + message history |
| DELETE | `/api/chat/sessions/{id}` | Delete a session |
| POST | `/api/chat` | Non-streaming chat (full answer at once) |
| POST | `/api/chat/stream` | Streaming chat (SSE) |

Full interactive docs available at `/api/docs` once the server is running.

---

## Testing

### Retrieval quality
```bash
cd backend && source venv/bin/activate
python -m tests.test_retrieval "What is the main topic of this document?"
```

### Full chat pipeline (no server needed)
```bash
python -m tests.test_chat "What does this document say about pricing?"
```

### Manual API test (server must be running)
```bash
# 1. Create a session
SESSION=$(curl -s -X POST http://localhost:8000/api/chat/sessions \
  -H "Content-Type: application/json" -d '{}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# 2. Non-streaming answer
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"question\": \"What is this document about?\"}" \
  | python3 -m json.tool

# 3. Streaming answer (tokens printed as they arrive)
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"question\": \"Give me more detail.\"}"
```

---

## Project Structure

```
pdf-chatbot/
├── Dockerfile                  # Multi-stage production image
├── docker-compose.yml          # Local dev with named volumes
├── render.yaml                 # Render deployment blueprint
├── .gitignore
├── README.md
├── backend/
│   ├── .env.example            # Config template (copy to .env)
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py             # FastAPI app, middleware, startup log
│   │   ├── config.py           # All settings (env-driven)
│   │   ├── models/
│   │   │   └── schemas.py      # Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── documents.py    # Upload, list, search, delete
│   │   │   └── chat.py         # Sessions, streaming chat
│   │   ├── services/
│   │   │   ├── pdf_processor.py   # PyMuPDF extraction
│   │   │   ├── chunking.py        # Page-aware overlapping chunks
│   │   │   ├── embeddings.py      # OpenAI batch embeddings
│   │   │   ├── vector_store.py    # ChromaDB + hybrid retrieval
│   │   │   ├── document_store.py  # JSON registry + pages JSON
│   │   │   └── chat.py            # Session store, RAG, SSE streaming
│   │   └── utils/
│   │       └── file_utils.py      # Validation, save, sanitize
│   ├── storage/
│   │   ├── uploads/            # Raw PDFs + page JSON (gitignored)
│   │   └── chroma/             # ChromaDB index (gitignored)
│   └── tests/
│       ├── test_retrieval.py   # CLI: test vector search quality
│       └── test_chat.py        # CLI: test full RAG pipeline
└── frontend/
    ├── templates/
    │   └── index.html          # Single-page app shell
    └── static/
        ├── app.js              # Upload, sessions, SSE streaming, sources panel
        └── style.css           # Dark theme, responsive layout
```

---

## License

MIT
