"""
Manual chat quality test — Phase 3.

Runs a full RAG pipeline directly (no HTTP server needed):
  1. Checks that at least one document is indexed in ChromaDB.
  2. Creates a chat session.
  3. Sends the question you supply and prints the answer + cited sources.
  4. Sends a follow-up question to verify conversation memory.

Usage (from the backend/ directory, venv active):

    python -m tests.test_chat "What is this document about?"

Requirements:
  - At least one PDF uploaded (so ChromaDB has chunks to search)
  - OPENAI_API_KEY set in backend/.env (for embeddings + chat if LLM_PROVIDER=openai)
  - Or ANTHROPIC_API_KEY if LLM_PROVIDER=anthropic
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import vector_store, chat as chat_service  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m tests.test_chat "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    # Sanity: check the vector store has data
    total = vector_store.collection_count()
    print(f"Vector store: {total} chunks indexed.\n")
    if total == 0:
        print("No chunks indexed. Upload a PDF first via POST /api/documents/upload.")
        sys.exit(1)

    # Create session
    session_id = chat_service.create_session(document_ids=[])
    print(f"Session: {session_id}\n")

    # First question
    print(f"Q1: {question}")
    print("-" * 60)
    result = chat_service.answer_question(session_id=session_id, question=question)
    print(f"A:  {result['answer']}\n")
    print(f"Sources ({len(result['sources'])}):")
    for s in result["sources"]:
        pages = ", ".join(str(p) for p in s["page_numbers"])
        print(f"  • {s['filename']}  page(s) {pages}  sim={s['similarity']:.3f}")
        print(f"    {s['excerpt'][:120]}…")
    print()

    # Follow-up (tests conversation memory)
    follow_up = "Can you elaborate on that?"
    print(f"Q2 (follow-up): {follow_up}")
    print("-" * 60)
    result2 = chat_service.answer_question(session_id=session_id, question=follow_up)
    print(f"A:  {result2['answer']}\n")
    print(f"Turn count: {result2['turn']}")

    # Cleanup
    chat_service.delete_session(session_id)
    print(f"\nSession deleted.")


if __name__ == "__main__":
    main()
