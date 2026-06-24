"""
Embeddings service.

Model choice: OpenAI `text-embedding-3-small`
  - Strong quality-to-cost ratio for general-purpose document retrieval.
  - 1536-dimensional vectors; fast to compute and store.
  - Kept separate from the chat LLM provider: even if LLM_PROVIDER=anthropic, embeddings
    still go through OpenAI, since Anthropic does not currently offer an embeddings API.
"""
from typing import List
from openai import OpenAI
from app.config import settings

_client = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Embeddings require an OpenAI API key "
                "even if you're using Anthropic for chat. Set it in backend/.env"
            )
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def embed_texts(texts: List[str], batch_size: int = 96) -> List[List[float]]:
    """Embed a list of texts in batches. Returns a list of embedding vectors in the
    same order as the input texts."""
    if not texts:
        return []

    client = get_openai_client()
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=batch)
        # Preserve order: response.data is returned in the same order as input
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
