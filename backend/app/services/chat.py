"""
Chat service — Phase 3.

Responsibilities:
  1. Session / conversation memory management (in-process, keyed by session_id).
  2. Retrieval: call vector_store.retrieve() with the user's question.
  3. Prompt assembly: build a system + user prompt grounding the LLM in retrieved chunks.
  4. LLM calling: support both OpenAI (gpt-4o-mini default) and Anthropic (claude-*).
  5. Streaming: yield text tokens as a generator (used by the SSE endpoint).
  6. Source formatting: return structured source attribution alongside the answer.

Prompt design
-------------
We use a strict RAG (Retrieval-Augmented Generation) prompt pattern:
  - The system prompt instructs the model to answer ONLY from provided context,
    cite page numbers, and say "I don't know" if the answer isn't in the context.
    This prevents hallucination and meets the source-attribution requirement.
  - Retrieved chunks are injected as numbered <source> blocks in the user message,
    each tagged with filename and page(s).
  - Chat history (last N turns) is prepended so the model can handle follow-up
    questions without losing context.

Conversation memory
-------------------
Sessions are stored in a plain dict in process memory, keyed by a UUID session_id
generated client-side.  Each session holds:
  - messages: List of {role, content} dicts — the raw conversation history.
  - document_ids: Optional list of document_ids to scope retrieval.
We keep at most MAX_HISTORY_TURNS pairs to avoid overflowing the context window.
"""

import uuid
import json
from typing import Iterator, Optional, List, Dict, Any

from app.config import settings
from app.services import vector_store

# ── Session store ─────────────────────────────────────────────────────────────

MAX_HISTORY_TURNS = 6  # keep last 6 user/assistant pairs

_sessions: Dict[str, Dict[str, Any]] = {}


def create_session(document_ids: Optional[List[str]] = None) -> str:
    session_id = uuid.uuid4().hex
    _sessions[session_id] = {
        "messages": [],
        "document_ids": document_ids or [],
    }
    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    return _sessions.get(session_id)


def delete_session(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def list_sessions() -> List[Dict[str, Any]]:
    return [
        {
            "session_id": sid,
            "turn_count": len(s["messages"]) // 2,
            "document_ids": s["document_ids"],
        }
        for sid, s in _sessions.items()
    ]


def _trim_history(messages: List[Dict]) -> List[Dict]:
    """Keep only the last MAX_HISTORY_TURNS user/assistant pairs."""
    # Each turn = 2 messages (user + assistant)
    max_msgs = MAX_HISTORY_TURNS * 2
    return messages[-max_msgs:] if len(messages) > max_msgs else messages


# ── Prompt assembly ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about uploaded PDF documents.

STRICT RULES:
1. Answer ONLY using the information provided in the <sources> block below.
2. If the answer cannot be found in the sources, say: "I don't have enough information in the provided documents to answer that question."
3. Always cite your sources. After each factual statement, include the citation in the format: [filename, page X] or [filename, pages X, Y].
4. Be concise and direct. Do not pad the answer.
5. If the user's question references previous messages, use both the conversation history AND the sources to answer.
"""


def _format_sources_block(chunks: List[Dict]) -> str:
    """Format retrieved chunks as a numbered <sources> block for injection into the prompt."""
    if not chunks:
        return "<sources>\n(No relevant sources found.)\n</sources>"

    lines = ["<sources>"]
    for i, chunk in enumerate(chunks, start=1):
        pages = ", ".join(str(p) for p in chunk["page_numbers"])
        page_label = f"page {pages}" if "," not in pages else f"pages {pages}"
        lines.append(
            f'<source index="{i}" file="{chunk["filename"]}" {page_label}>\n'
            f'{chunk["text"]}\n'
            f"</source>"
        )
    lines.append("</sources>")
    return "\n".join(lines)


def _build_openai_messages(
    history: List[Dict], sources_block: str, user_question: str
) -> List[Dict]:
    """Assemble the messages list for the OpenAI chat completions API."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Previous turns (already trimmed)
    messages.extend(history)
    # New user message with sources injected
    messages.append(
        {
            "role": "user",
            "content": f"{sources_block}\n\nQuestion: {user_question}",
        }
    )
    return messages


def _build_anthropic_messages(
    history: List[Dict], sources_block: str, user_question: str
) -> tuple[str, List[Dict]]:
    """Assemble system prompt + messages list for the Anthropic Messages API."""
    # Anthropic keeps system prompt separate from messages
    messages = list(history)
    messages.append(
        {
            "role": "user",
            "content": f"{sources_block}\n\nQuestion: {user_question}",
        }
    )
    return SYSTEM_PROMPT, messages


# ── LLM calling ──────────────────────────────────────────────────────────────

def _stream_openai(messages: List[Dict]) -> Iterator[str]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    with client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=messages,
        stream=True,
        max_tokens=1024,
        temperature=0.2,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def _stream_anthropic(system: str, messages: List[Dict]) -> Iterator[str]:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    with client.messages.stream(
        model=settings.ANTHROPIC_CHAT_MODEL,
        system=system,
        messages=messages,
        max_tokens=1024,
    ) as stream:
        for text in stream.text_stream:
            yield text


def _call_openai_sync(messages: List[Dict]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def _call_anthropic_sync(system: str, messages: List[Dict]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=settings.ANTHROPIC_CHAT_MODEL,
        system=system,
        messages=messages,
        max_tokens=1024,
    )
    return resp.content[0].text


# ── Public API ────────────────────────────────────────────────────────────────

def _validate_provider():
    provider = settings.LLM_PROVIDER.lower()
    if provider == "openai" and not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to backend/.env")
    if provider == "anthropic" and not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to backend/.env")
    if provider not in ("openai", "anthropic"):
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider!r}. Use 'openai' or 'anthropic'.")


def answer_question(
    session_id: str,
    question: str,
) -> Dict[str, Any]:
    """
    Non-streaming answer.  Returns:
      { answer: str, sources: List[source_dict], session_id: str, turn: int }
    """
    _validate_provider()

    session = _sessions.get(session_id)
    if session is None:
        raise ValueError(f"Session {session_id!r} not found.")

    # Retrieve relevant chunks
    chunks = vector_store.retrieve(
        query=question,
        top_k=settings.TOP_K,
        document_ids=session["document_ids"] or None,
    )

    history = _trim_history(session["messages"])
    sources_block = _format_sources_block(chunks)

    provider = settings.LLM_PROVIDER.lower()
    if provider == "openai":
        openai_messages = _build_openai_messages(history, sources_block, question)
        answer = _call_openai_sync(openai_messages)
    else:
        system, anth_messages = _build_anthropic_messages(history, sources_block, question)
        answer = _call_anthropic_sync(system, anth_messages)

    # Persist this turn in history
    session["messages"].append({"role": "user", "content": question})
    session["messages"].append({"role": "assistant", "content": answer})

    # Build structured source list for the response
    sources = _dedupe_sources(chunks)

    return {
        "answer": answer,
        "sources": sources,
        "session_id": session_id,
        "turn": len(session["messages"]) // 2,
    }


def stream_answer(
    session_id: str,
    question: str,
) -> Iterator[str]:
    """
    Streaming answer as a generator of SSE-formatted strings.

    Yields lines formatted as Server-Sent Events:
      data: <json>\n\n

    Event types:
      { type: "source", payload: <source_list> }   — sent first, before tokens
      { type: "token",  payload: "<text>" }         — one per LLM token
      { type: "done",   payload: { turn: N } }      — final event
      { type: "error",  payload: "<message>" }      — on failure

    The client accumulates "token" payloads to build the full answer, and
    displays "source" immediately so citations appear while the answer streams.
    """
    _validate_provider()

    session = _sessions.get(session_id)
    if session is None:
        yield _sse({"type": "error", "payload": f"Session {session_id!r} not found."})
        return

    # Retrieve
    try:
        chunks = vector_store.retrieve(
            query=question,
            top_k=settings.TOP_K,
            document_ids=session["document_ids"] or None,
        )
    except Exception as e:
        yield _sse({"type": "error", "payload": f"Retrieval failed: {e}"})
        return

    # Emit sources immediately (before LLM starts)
    sources = _dedupe_sources(chunks)
    yield _sse({"type": "sources", "payload": sources})

    history = _trim_history(session["messages"])
    sources_block = _format_sources_block(chunks)
    full_answer = []

    try:
        provider = settings.LLM_PROVIDER.lower()
        if provider == "openai":
            openai_messages = _build_openai_messages(history, sources_block, question)
            for token in _stream_openai(openai_messages):
                full_answer.append(token)
                yield _sse({"type": "token", "payload": token})
        else:
            system, anth_messages = _build_anthropic_messages(history, sources_block, question)
            for token in _stream_anthropic(system, anth_messages):
                full_answer.append(token)
                yield _sse({"type": "token", "payload": token})
    except Exception as e:
        yield _sse({"type": "error", "payload": f"LLM error: {e}"})
        return

    # Persist completed turn
    answer_text = "".join(full_answer)
    session["messages"].append({"role": "user", "content": question})
    session["messages"].append({"role": "assistant", "content": answer_text})

    yield _sse({"type": "done", "payload": {"turn": len(session["messages"]) // 2}})


def _sse(data: Dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _dedupe_sources(chunks: List[Dict]) -> List[Dict]:
    """Build a clean, deduplicated source list for the API response."""
    seen = set()
    sources = []
    for c in chunks:
        key = (c["document_id"], tuple(c["page_numbers"]))
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "document_id": c["document_id"],
                    "filename": c["filename"],
                    "page_numbers": c["page_numbers"],
                    "excerpt": c["text"][:300],
                    "similarity": c["similarity"],
                }
            )
    return sources
