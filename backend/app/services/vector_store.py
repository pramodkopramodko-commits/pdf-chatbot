"""
Vector store service (ChromaDB).

Retrieval approach
------------------
- Persistent local ChromaDB collection (cosine similarity over OpenAI embeddings).
- Each chunk is stored with metadata: document_id, filename, page_numbers (as a
  comma-separated string, since Chroma metadata values must be scalars), chunk_id.
- Querying: embed the user's question, run a top-k similarity search, optionally
  filtered to specific document_ids (for multi-PDF sessions where a user might want
  to scope to one document).
- Hybrid search (bonus): a lightweight keyword overlap re-ranking pass on top of the
  vector results, to catch exact-term matches (e.g. acronyms, names) that embedding
  similarity sometimes under-ranks.
"""
import chromadb
from typing import List, Optional
from app.config import settings
from app.services.chunking import Chunk
from app.services.embeddings import embed_texts, embed_query

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def index_chunks(chunks: List[Chunk], document_id: str, filename: str) -> int:
    """Embed and store a document's chunks in the vector store. Returns count indexed."""
    if not chunks:
        return 0

    collection = get_collection()
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "document_id": document_id,
            "filename": filename,
            "page_numbers": ",".join(str(p) for p in c.page_numbers),
            "char_start": c.char_start,
            "char_end": c.char_end,
        }
        for c in chunks
    ]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)


def delete_document_chunks(document_id: str) -> None:
    collection = get_collection()
    try:
        collection.delete(where={"document_id": document_id})
    except Exception:
        pass  # collection may be empty / no matching chunks


def _keyword_overlap_score(query: str, text: str) -> int:
    """Simple keyword overlap count used for hybrid re-ranking."""
    query_terms = {t.lower() for t in query.split() if len(t) > 2}
    text_lower = text.lower()
    return sum(1 for term in query_terms if term in text_lower)


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    document_ids: Optional[List[str]] = None,
    use_hybrid: bool = True,
) -> List[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Returns a list of dicts: {text, document_id, filename, page_numbers (List[int]), score}
    sorted by relevance (best first).
    """
    collection = get_collection()
    k = top_k or settings.TOP_K

    query_embedding = embed_query(query)

    where_filter = None
    if document_ids:
        where_filter = {"document_id": {"$in": document_ids}}

    # Over-fetch a bit when hybrid re-ranking is enabled, then re-rank and trim to k.
    n_fetch = k * 3 if use_hybrid else k

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_fetch,
        where=where_filter,
    )

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, distance in zip(docs, metas, distances):
        similarity = 1 - distance  # cosine distance -> similarity
        keyword_score = _keyword_overlap_score(query, text) if use_hybrid else 0
        hits.append(
            {
                "text": text,
                "document_id": meta.get("document_id"),
                "filename": meta.get("filename"),
                "page_numbers": [int(p) for p in meta.get("page_numbers", "").split(",") if p],
                "similarity": round(similarity, 4),
                "keyword_score": keyword_score,
            }
        )

    if use_hybrid:
        # Combine vector similarity with keyword overlap (small weight) for re-ranking.
        hits.sort(key=lambda h: (h["similarity"] + 0.03 * h["keyword_score"]), reverse=True)

    return hits[:k]


def collection_count() -> int:
    return get_collection().count()
