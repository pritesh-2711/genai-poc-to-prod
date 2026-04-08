"""Chat message endpoints.

GET  /sessions/{session_id}/messages — fetch full conversation history
POST /sessions/{session_id}/messages — send a message and get the LLM reply
POST /sessions/{session_id}/messages/stream — same, streamed via SSE

POST flow (non-streaming):
  1. Embed query once  →  persist user message with embedding
  2. Check if session has a pending clarification thread (deep mode HITL)
     a. If yes  → resume interrupted graph with user message as clarification
     b. If no   → start fresh graph run (fast or deep based on body.mode)
  3. Graph runs the full RAG cycle (memory, retrieval, rerank, generate, [validate])
  4. If graph is interrupted (query_clarification) →
       return clarification question as assistant reply + store pending thread
  5. If graph completes → persist assistant reply + return both records
"""

import json
import logging
import uuid
from typing import Annotated, AsyncGenerator, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from ..core.exceptions import InputBlockedError
from ..core.models import UserRecord
from ..embedding.base import BaseEmbedder
from ..memory.repository import MemoryRepository, MemoryRepositoryError
from ..orchestrators import RAGOrchestrator
from ..orchestrators.state import RAGState
from .deps import (
    get_current_user,
    get_embedder,
    get_orchestrator,
    get_pending_clarifications,
    get_repo,
)
from .schemas import ChatMessageResponse, SendMessageRequest, SendMessageResponse

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

_BLOCKED_REPLY = (
    "I'm sorry, but I'm not able to respond to that message. "
    "Please keep our conversation respectful and on-topic."
)


def _to_msg_response(record) -> ChatMessageResponse:
    return ChatMessageResponse(
        chat_id=record.chat_id,
        session_id=record.session_id,
        sender=record.sender,
        message=record.message,
        created_at=record.created_at,
    )


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
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    repo: Annotated[MemoryRepository, Depends(get_repo)],
    embedder: Annotated[BaseEmbedder, Depends(get_embedder)],
    orchestrator: Annotated[RAGOrchestrator, Depends(get_orchestrator)],
    pending_clarifications: Annotated[dict, Depends(get_pending_clarifications)],
):
    """Persist the user message, run the RAG orchestrator, persist reply.

    Handles both normal turns and deep-mode clarification resumes transparently.
    """
    session_id_str = str(session_id)
    user_id_str = str(current_user.user_id)

    try:
        # ── 1. Embed query + persist user message ────────────────────────────
        query_vec = embedder.embed_query(body.message)
        user_record = repo.add_message(
            session_id=session_id,
            sender="user",
            message=body.message,
            embedding=query_vec,
        )
        user_chat_id_str = str(user_record.chat_id)

        # ── 2. Determine whether to resume or start fresh ────────────────────
        pending_thread_id = pending_clarifications.get(session_id_str)

        if pending_thread_id and orchestrator.is_interrupted(pending_thread_id):
            # Resume the paused deep-mode graph with the user's clarification
            logger.info(
                f"[chat] session={session_id} | resuming clarification thread={pending_thread_id}"
            )
            result: RAGState = await orchestrator.aresume(
                thread_id=pending_thread_id,
                user_clarification=body.message,
            )
            # Remove from pending — thread is now resolved
            pending_clarifications.pop(session_id_str, None)

        else:
            # Start a fresh graph run with a new thread_id per turn
            thread_id = user_chat_id_str
            initial_state: RAGState = {
                "original_query": body.message,
                "query_embedding": query_vec,
                "mode": body.mode,
                "session_id": session_id_str,
                "user_id": user_id_str,
                "user_chat_id": user_chat_id_str,
                # Accumulator fields initialised to empty so operator.add works
                "raw_chunks": [],
                "iteration_count": 0,
                "best_response": "",
                "validation_result": "",
                "correction_note": "",
            }

            result = await orchestrator.ainvoke(initial_state, thread_id=thread_id)

            # Check if the graph paused for clarification (deep mode only)
            if orchestrator.is_interrupted(thread_id):
                clarification_q = (
                    result.get("clarification_question")
                    or orchestrator.get_clarification_question(thread_id)
                    or "Could you clarify your question?"
                )
                logger.info(
                    f"[chat] session={session_id} | graph interrupted, asking: {clarification_q!r}"
                )
                # Store the pending thread so the next request can resume it
                pending_clarifications[session_id_str] = thread_id

                # Persist the clarification question as the assistant reply
                assistant_record = repo.add_message(
                    session_id=session_id,
                    sender="assistant",
                    message=clarification_q,
                )
                return SendMessageResponse(
                    user_message=_to_msg_response(user_record),
                    assistant_message=_to_msg_response(assistant_record),
                )

        # ── 3. Graph completed — persist assistant reply ──────────────────────
        assistant_text = (
            result.get("final_response")
            or result.get("llm_response")
            or ""
        )
        assistant_vec = embedder.embed_one(assistant_text)
        assistant_record = repo.add_message(
            session_id=session_id,
            sender="assistant",
            message=assistant_text,
            embedding=assistant_vec,
            metadata={
                "mode": body.mode,
                "query_complexity": result.get("query_complexity", ""),
                "iterations": result.get("iteration_count", 0),
                "validation_result": result.get("validation_result", ""),
            },
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
        logger.exception("Unexpected error in send_message")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Orchestrator error: {e}",
        )

    return SendMessageResponse(
        user_message=_to_msg_response(user_record),
        assistant_message=_to_msg_response(assistant_record),
    )


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Events data line."""
    return f"data: {json.dumps(data, default=str)}\n\n"


# ── Deep-mode node → human-readable status message ───────────────────────────
# Only non-None values are emitted. Identical consecutive statuses are
# deduplicated (e.g., retrieve_sub_query fires N times for decomposed queries).
_DEEP_NODE_STATUS: dict[str, str | None] = {
    "resolve_memory":          "Loading conversation history…",
    "analyze_query":           "Checking query intent…",
    "route_complexity":        "Identifying complexity…",
    "query_clarification":     None,   # handled as interrupt; no status needed
    "query_rewrite":           "Optimising query for retrieval…",
    "query_decompose":         "Breaking down your question…",
    "retrieve":                "Searching documents…",
    "retrieve_sub_query":      "Searching documents…",
    "rerank_and_build_context": "Ranking relevant results…",
    "generate":                "Generating response…",
    "validate_response":       "Validating answer quality…",
    "correction":              "Refining the answer…",
    "finalize":                None,
}


@router.post("/sessions/{session_id}/messages/stream")
async def stream_message(
    session_id: UUID,
    body: SendMessageRequest,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    repo: Annotated[MemoryRepository, Depends(get_repo)],
    embedder: Annotated[BaseEmbedder, Depends(get_embedder)],
    orchestrator: Annotated[RAGOrchestrator, Depends(get_orchestrator)],
    pending_clarifications: Annotated[dict, Depends(get_pending_clarifications)],
):
    """Same RAG cycle as send_message, but streams the LLM reply token-by-token via SSE.

    SSE event types:
      {"type": "user_message", ...}
      {"type": "status",  "content": "Searching documents..."}   (graph progress)
      {"type": "token",   "content": "<chunk>"}                  (LLM tokens)
      {"type": "clarification", "content": "<question>"}         (deep mode HITL)
      {"type": "done",    ...assistant record...}
      {"type": "error",   "detail": "..."}
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        session_id_str = str(session_id)
        user_id_str = str(current_user.user_id)

        try:
            # ── 1. Embed + persist user message ──────────────────────────────
            query_vec = embedder.embed_query(body.message)
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

            user_chat_id_str = str(user_record.chat_id)
            pending_thread_id = pending_clarifications.get(session_id_str)
            result: Optional[RAGState] = None

            # ── 2. Resume interrupted thread (deep mode clarification) ────────
            if pending_thread_id and orchestrator.is_interrupted(pending_thread_id):
                yield _sse({"type": "status", "content": "Resuming after clarification…"})
                result = await orchestrator.aresume(
                    thread_id=pending_thread_id,
                    user_clarification=body.message,
                )
                pending_clarifications.pop(session_id_str, None)

            else:
                # ── 3. Fresh graph run — stream per-node status events ────────
                thread_id = user_chat_id_str
                initial_state: RAGState = {
                    "original_query": body.message,
                    "query_embedding": query_vec,
                    "mode": body.mode,
                    "session_id": session_id_str,
                    "user_id": user_id_str,
                    "user_chat_id": user_chat_id_str,
                    "raw_chunks": [],
                    "iteration_count": 0,
                    "best_response": "",
                    "validation_result": "",
                    "correction_note": "",
                }

                last_status: str | None = None
                is_deep = body.mode == "deep"

                async for chunk in orchestrator.astream_updates(initial_state, thread_id=thread_id):
                    # chunk = (namespace_tuple, {node_name: state_delta})
                    namespace, update = chunk
                    if not namespace:
                        # Outer graph node completed (e.g. fast_graph / deep_graph wrapper)
                        continue

                    node_name = next(iter(update))  # the completing node

                    if is_deep:
                        status_msg = _DEEP_NODE_STATUS.get(node_name)
                        if status_msg and status_msg != last_status:
                            yield _sse({"type": "status", "content": status_msg})
                            last_status = status_msg

                # ── 4. Retrieve final state after stream ──────────────────────
                graph_state = orchestrator.get_graph_state(thread_id)

                if graph_state and graph_state.next:
                    # Graph is interrupted — deep mode awaiting clarification
                    clarification_q = (
                        graph_state.values.get("clarification_question")
                        or orchestrator.get_clarification_question(thread_id)
                        or "Could you clarify your question?"
                    )
                    pending_clarifications[session_id_str] = thread_id

                    assistant_record = repo.add_message(
                        session_id=session_id,
                        sender="assistant",
                        message=clarification_q,
                    )
                    yield _sse({"type": "clarification", "content": clarification_q})
                    yield _sse({
                        "type": "done",
                        **_to_msg_response(assistant_record).model_dump(mode="json"),
                    })
                    return

                result = graph_state.values if graph_state else {}

            # ── 5. Emit final response ────────────────────────────────────────
            assistant_text = (
                (result or {}).get("final_response")
                or (result or {}).get("llm_response")
                or ""
            )

            for word in assistant_text.split(" "):
                yield _sse({"type": "token", "content": word + " "})

            # ── 6. Persist assistant reply + emit done ────────────────────────
            try:
                assistant_vec = embedder.embed_one(assistant_text)
                assistant_record = repo.add_message(
                    session_id=session_id,
                    sender="assistant",
                    message=assistant_text,
                    embedding=assistant_vec,
                    metadata={
                        "mode": body.mode,
                        "query_complexity": (result or {}).get("query_complexity", ""),
                        "iterations": (result or {}).get("iteration_count", 0),
                        "validation_result": (result or {}).get("validation_result", ""),
                    },
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
