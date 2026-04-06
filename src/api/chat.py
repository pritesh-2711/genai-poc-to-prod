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

import logging
from typing import Annotated, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

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


def _to_msg_response(record) -> ChatMessageResponse:
    return ChatMessageResponse(
        chat_id=record.chat_id,
        session_id=record.session_id,
        sender=record.sender,
        message=record.message,
        created_at=record.created_at,
    )


async def _build_rag_context(
    query: str,
    session_id: UUID,
    embedder: BaseEmbedder,
    retrieval_repo: PgVectorRetrievalRepository,
) -> Optional[str]:
    """Embed the query, retrieve top-K child chunks, and fetch their parent contexts.

    Returns a formatted context string, or None if retrieval is unavailable or
    no results are found.
    """
    try:
        query_vec = embedder.embed_query(query)
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
        user_record = repo.add_message(
            session_id=session_id,
            sender="user",
            message=body.message,
        )

        # Retrieve grounding context from uploaded documents (best-effort)
        rag_context = await _build_rag_context(
            query=body.message,
            session_id=session_id,
            embedder=embedder,
            retrieval_repo=retrieval_repo,
        )

        history = repo.get_conversation_history(session_id=session_id)
        context_history = history[:-1]  # exclude the message just added

        assistant_text = await chat_service.get_response_async(
            user_message=body.message,
            history=context_history,
            rag_context=rag_context,
        )

        assistant_record = repo.add_message(
            session_id=session_id,
            sender="assistant",
            message=assistant_text,
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
