"""Chat message endpoints.

GET  /sessions/{session_id}/messages — fetch full conversation history
POST /sessions/{session_id}/messages — send a message and get the LLM reply

The POST endpoint runs the full RAG cycle in one call:
  1. Persist the user message
  2. Embed the query and retrieve the top-K relevant chunks from uploaded docs
  3. Fetch parent context for the matched child chunks
  4. Build a RAG context block and inject it into the system prompt
  5. Fetch conversation history for the session
  6. Call ChatService.get_response_async (grounded by retrieved context)
  7. Persist the assistant reply
  8. Return both records to the frontend

Retrieval is best-effort: if no documents are uploaded for the session or
the vector search fails, the assistant responds from its base knowledge.
"""

import json
import logging
from typing import Annotated, AsyncGenerator, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from ..chat_service import ChatService
from ..core.exceptions import InputBlockedError
from ..core.models import UserRecord
from ..databases.retrieval import PgVectorRetrievalRepository
from ..embedding.base import BaseEmbedder
from ..memory.repository import MemoryRepository, MemoryRepositoryError
from .deps import get_chat_service, get_current_user, get_embedder, get_repo, get_retrieval_repo
from .schemas import ChatMessageResponse, SendMessageRequest, SendMessageResponse

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

_BLOCKED_REPLY = (
    "I'm sorry, but I'm not able to respond to that message. "
    "Please keep our conversation respectful and on-topic."
)

_RAG_TOP_K = 10
_MAX_CONTEXT_CHARS = 8000  # cap total context injected into system prompt


_LONG_TERM_TOP_K = 10


def _to_msg_response(record) -> ChatMessageResponse:
    return ChatMessageResponse(
        chat_id=record.chat_id,
        session_id=record.session_id,
        sender=record.sender,
        message=record.message,
        created_at=record.created_at,
    )


async def _resolve_memory(
    query_vec: list[float],
    session_id: UUID,
    user_chat_id: str,
    repo: MemoryRepository,
    retrieval_repo: PgVectorRetrievalRepository,
    short_term_limit: int,
    similarity_threshold: float,
) -> tuple[list, list]:
    """Build short-term and long-term memory for the current turn.

    Short-term: last ``short_term_limit`` messages (chronological, excluding the
    just-persisted user message).

    Long-term: cosine-similarity search over embedded chat history, but ONLY
    when the session has already crossed ``short_term_limit`` messages — i.e.
    there is history beyond what short-term already covers. Results below
    ``similarity_threshold`` are dropped. Duplicates already in short-term
    are removed.

    Returns:
        (short_term_history, long_term_history)
    """
    # Fetch one extra row so we can detect whether the session has crossed the
    # short_term_limit without a separate COUNT query.
    raw_history = repo.get_conversation_history(
        session_id=session_id, limit=short_term_limit + 1
    )
    short_term_history = raw_history[:-1]  # drop the just-added user message

    logger.info(
        f"[memory] session={session_id} | short-term fetched: {len(short_term_history)} "
        f"(limit={short_term_limit})"
    )

    # Guard: activate long-term only when session history exceeds short_term_limit.
    # raw_history contains at most short_term_limit+1 rows. If it is full
    # (== short_term_limit+1), there are guaranteed to be older messages that
    # short-term does not cover — long-term search is meaningful.
    session_exceeds_limit = len(raw_history) == short_term_limit + 1

    if not session_exceeds_limit:
        logger.info(
            f"[memory] session={session_id} | long-term skipped "
            f"(session has not yet crossed short_term_limit={short_term_limit})"
        )
        return short_term_history, []

    # Long-term semantic search
    long_term_raw = await retrieval_repo.search_conversation_history(
        query_embedding=query_vec,
        session_id=session_id,
        top_k=_LONG_TERM_TOP_K,
        exclude_chat_id=user_chat_id,
    )
    logger.info(
        f"[memory] session={session_id} | long-term raw results: {len(long_term_raw)}"
    )

    # Apply similarity threshold
    long_term_above_threshold = [
        r for r in long_term_raw if r["similarity"] >= similarity_threshold
    ]
    logger.info(
        f"[memory] session={session_id} | long-term after threshold "
        f"(>= {similarity_threshold}): {len(long_term_above_threshold)}"
    )

    # Deduplicate against short-term
    short_term_ids = {str(r.chat_id) for r in short_term_history}
    long_term_history = [
        r for r in long_term_above_threshold if r["chat_id"] not in short_term_ids
    ]
    dropped = len(long_term_above_threshold) - len(long_term_history)
    logger.info(
        f"[memory] session={session_id} | long-term after dedup: {len(long_term_history)} "
        f"({dropped} duplicate(s) removed)"
    )

    return short_term_history, long_term_history


async def _build_rag_context(
    query_vec: list[float],
    session_id: UUID,
    retrieval_repo: PgVectorRetrievalRepository,
) -> Optional[str]:
    """Retrieve top-K child chunks using a pre-computed query embedding and fetch
    their parent contexts.

    Accepts an already-computed query embedding so the caller can embed once
    and reuse the vector for both RAG retrieval and long-term memory search.

    Returns a formatted context string, or None if retrieval is unavailable or
    no results are found.
    """
    try:
        child_hits = await retrieval_repo.search(
            query_embedding=query_vec,
            top_k=_RAG_TOP_K,
            session_id=session_id,
        )

        if not child_hits:
            return None

        # Collect unique parent UUIDs
        parent_ids = list({
            hit["parent_id"]
            for hit in child_hits
            if hit.get("parent_id")
        })

        # Prefer parent chunks (richer context); fall back to child text
        if parent_ids:
            parents = await retrieval_repo.fetch_parent_contexts(parent_ids)
        else:
            parents = []

        if not parents and not child_hits:
            return None

        # ----------------------------------------------------------------
        # Co-located retrieval: fetch table/image chunks from the same
        # pages as the matched parent contexts, regardless of their own
        # vector similarity score.
        # ----------------------------------------------------------------
        colocated: list[dict] = []
        if parents:
            # Collect page numbers and filenames from parent metadata
            pages_seen: set[int] = set()
            filenames_seen: set[str] = set()
            for p in parents:
                meta = p.get("metadata") or {}
                page = meta.get("page")
                if isinstance(page, int):
                    pages_seen.add(page)
                if p.get("filename"):
                    filenames_seen.add(p["filename"])

            if pages_seen and filenames_seen:
                colocated = await retrieval_repo.fetch_colocated_chunks(
                    session_id=session_id,
                    pages=list(pages_seen),
                    filenames=list(filenames_seen),
                )

        # ----------------------------------------------------------------
        # Build context block
        # ----------------------------------------------------------------
        context_parts: list[str] = []
        total = 0

        # Primary passages: parent text chunks
        passages = [p["parent_chunk_content"] for p in parents if p.get("parent_chunk_content")]
        if not passages:
            passages = [hit["chunk_content"] for hit in child_hits]

        for i, passage in enumerate(passages, 1):
            snippet = passage.strip()
            if total + len(snippet) > _MAX_CONTEXT_CHARS:
                snippet = snippet[: _MAX_CONTEXT_CHARS - total]
            context_parts.append(f"[{i}] {snippet}")
            total += len(snippet)
            if total >= _MAX_CONTEXT_CHARS:
                break

        # Append co-located table/image chunks (deduplicated by content)
        seen_content: set[str] = set()
        for chunk in colocated:
            if total >= _MAX_CONTEXT_CHARS:
                break
            text = chunk["chunk_content"].strip()
            if not text or text in seen_content:
                continue
            seen_content.add(text)
            ctype = chunk.get("content_type", "table")
            page = (chunk.get("metadata") or {}).get("page", "?")
            label = f"[{ctype.capitalize()} p.{page}]"
            snippet = f"{label} {text}"
            if total + len(snippet) > _MAX_CONTEXT_CHARS:
                snippet = snippet[: _MAX_CONTEXT_CHARS - total]
            context_parts.append(snippet)
            total += len(snippet)

        if not context_parts:
            return None

        return "\n\n".join(context_parts)

    except Exception as exc:
        logger.warning(f"RAG retrieval failed (proceeding without context): {exc}")
        return None


@router.get(
    "/sessions/{session_id}/messages",
    response_model=List[ChatMessageResponse],
)
def get_messages(
    session_id: UUID,
    _current_user: Annotated[UserRecord, Depends(get_current_user)],
    repo: Annotated[MemoryRepository, Depends(get_repo)],
):
    """Return the full conversation history for a session, oldest first."""
    try:
        records = repo.get_conversation_history(session_id=session_id)
    except MemoryRepositoryError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return [_to_msg_response(r) for r in records]


@router.post(
    "/sessions/{session_id}/messages",
    response_model=SendMessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def send_message(
    session_id: UUID,
    body: SendMessageRequest,
    _current_user: Annotated[UserRecord, Depends(get_current_user)],
    repo: Annotated[MemoryRepository, Depends(get_repo)],
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
    embedder: Annotated[BaseEmbedder, Depends(get_embedder)],
    retrieval_repo: Annotated[PgVectorRetrievalRepository, Depends(get_retrieval_repo)],
):
    """Persist the user message, retrieve relevant document context, get LLM reply.

    RAG retrieval is session-scoped — only chunks from documents uploaded to
    this session are searched. If retrieval yields nothing the LLM responds
    from its own knowledge.
    """
    try:
        # 1. Embed query once — reused for RAG retrieval + long-term memory search
        query_vec = embedder.embed_query(body.message)

        # 2. Persist user message with its embedding
        user_record = repo.add_message(
            session_id=session_id,
            sender="user",
            message=body.message,
            embedding=query_vec,
        )

        # 3. RAG context (best-effort) — no re-embedding
        rag_context = await _build_rag_context(
            query_vec=query_vec,
            session_id=session_id,
            retrieval_repo=retrieval_repo,
        )

        # 4 & 5 & 6. Short-term + long-term memory (with guard, threshold, dedup)
        short_term_history, long_term_history = await _resolve_memory(
            query_vec=query_vec,
            session_id=session_id,
            user_chat_id=str(user_record.chat_id),
            repo=repo,
            retrieval_repo=retrieval_repo,
            short_term_limit=chat_service.chat_config.short_term_limit,
            similarity_threshold=chat_service.chat_config.long_term_similarity_threshold,
        )

        # 7. Generate LLM response
        assistant_text = await chat_service.get_response_async(
            user_message=body.message,
            short_term_history=short_term_history,
            long_term_history=long_term_history,
            rag_context=rag_context,
        )

        # 8. Persist assistant reply with its embedding
        assistant_vec = embedder.embed_one(assistant_text)
        assistant_record = repo.add_message(
            session_id=session_id,
            sender="assistant",
            message=assistant_text,
            embedding=assistant_vec,
        )

    except InputBlockedError:
        assistant_record = repo.add_message(
            session_id=session_id,
            sender="assistant",
            message=_BLOCKED_REPLY,
        )
    except MemoryRepositoryError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM error: {e}",
        )

    return SendMessageResponse(
        user_message=_to_msg_response(user_record),
        assistant_message=_to_msg_response(assistant_record),
    )


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Events data line."""
    return f"data: {json.dumps(data, default=str)}\n\n"


@router.post("/sessions/{session_id}/messages/stream")
async def stream_message(
    session_id: UUID,
    body: SendMessageRequest,
    _current_user: Annotated[UserRecord, Depends(get_current_user)],
    repo: Annotated[MemoryRepository, Depends(get_repo)],
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
    embedder: Annotated[BaseEmbedder, Depends(get_embedder)],
    retrieval_repo: Annotated[PgVectorRetrievalRepository, Depends(get_retrieval_repo)],
):
    """Same RAG cycle as send_message, but streams the LLM reply token-by-token via SSE.

    SSE event types:
      {"type": "user_message", "chat_id": ..., "session_id": ..., "sender": "user",
       "message": ..., "created_at": ...}
      {"type": "token", "content": "<chunk>"}   (repeated)
      {"type": "done",  "chat_id": ..., "session_id": ..., "sender": "assistant",
       "message": <full text>, "created_at": ...}
      {"type": "error", "detail": "..."}        (on failure)
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 1. Embed query once — reused for RAG + long-term memory search
            query_vec = embedder.embed_query(body.message)

            # 2. Persist user message with its embedding and emit immediately
            try:
                user_record = repo.add_message(
                    session_id=session_id,
                    sender="user",
                    message=body.message,
                    embedding=query_vec,
                )
            except MemoryRepositoryError as e:
                yield _sse({"type": "error", "detail": str(e)})
                return

            yield _sse({
                "type": "user_message",
                **_to_msg_response(user_record).model_dump(mode="json"),
            })

            # 3. RAG context (best-effort) — no re-embedding
            rag_context = await _build_rag_context(
                query_vec=query_vec,
                session_id=session_id,
                retrieval_repo=retrieval_repo,
            )

            # 4 & 5 & 6. Short-term + long-term memory (with guard, threshold, dedup)
            short_term_history, long_term_history = await _resolve_memory(
                query_vec=query_vec,
                session_id=session_id,
                user_chat_id=str(user_record.chat_id),
                repo=repo,
                retrieval_repo=retrieval_repo,
                short_term_limit=chat_service.chat_config.short_term_limit,
                similarity_threshold=chat_service.chat_config.long_term_similarity_threshold,
            )

            # 7. Stream LLM tokens
            full_text = ""
            try:
                async for chunk in chat_service.stream_response_async(
                    user_message=body.message,
                    short_term_history=short_term_history,
                    long_term_history=long_term_history,
                    rag_context=rag_context,
                ):
                    full_text += chunk
                    yield _sse({"type": "token", "content": chunk})
            except InputBlockedError:
                full_text = _BLOCKED_REPLY
                yield _sse({"type": "token", "content": full_text})

            # 8. Persist assistant reply with its embedding and emit done event
            try:
                assistant_vec = embedder.embed_one(full_text)
                assistant_record = repo.add_message(
                    session_id=session_id,
                    sender="assistant",
                    message=full_text,
                    embedding=assistant_vec,
                )
            except MemoryRepositoryError as e:
                yield _sse({"type": "error", "detail": str(e)})
                return

            yield _sse({
                "type": "done",
                **_to_msg_response(assistant_record).model_dump(mode="json"),
            })

        except Exception as e:
            logger.exception("Unexpected error in stream_message")
            yield _sse({"type": "error", "detail": f"Unexpected error: {e}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
