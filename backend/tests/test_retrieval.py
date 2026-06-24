"""
Manual retrieval quality test.

This is not a pytest suite (no CI/auto-grading needed for this assignment) — it's a
small CLI script you run after uploading at least one PDF, to sanity-check that:
  1. Chunks were created and embedded correctly.
  2. Querying returns relevant chunks.
  3. Page-number metadata is correctly attached (for source attribution in Phase 3).

Usage (from backend/ directory, with the server NOT required to be running --
this talks to ChromaDB directly):

    python -m tests.test_retrieval "What is the termination clause?"

Make sure you've already uploaded a PDF via the running server (or via
scripts that call the /api/documents/upload endpoint) before running this.
"""
import sys
from pathlib import Path

# Allow running as `python -m tests.test_retrieval` from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import vector_store  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m tests.test_retrieval "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    total = vector_store.collection_count()
    print(f"Vector store currently holds {total} chunks.\n")

    if total == 0:
        print("No chunks indexed yet. Upload a PDF first via POST /api/documents/upload.")
        sys.exit(1)

    print(f"Query: {query!r}\n")
    results = vector_store.retrieve(query, top_k=5)

    if not results:
        print("No results returned.")
        return

    for i, r in enumerate(results, start=1):
        pages = ", ".join(str(p) for p in r["page_numbers"])
        print(f"--- Result {i} ---")
        print(f"Document : {r['filename']}")
        print(f"Page(s)  : {pages}")
        print(f"Similarity: {r['similarity']}  | Keyword overlap: {r['keyword_score']}")
        excerpt = r["text"][:300].replace("\n", " ")
        print(f"Excerpt  : {excerpt}...")
        print()


if __name__ == "__main__":
    main()
