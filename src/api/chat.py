"""Chat message endpoints.

GET  /sessions/{session_id}/messages — fetch full conversation history
POST /sessions/{session_id}/messages — send a message and get the LLM reply

The POST endpoint runs the full cycle in one call:
  1. Persist the user message
  2. Fetch conversation history for context
  3. Call ChatService.get_response_async (Ollama or OpenAI via LangChain)
  4. Persist the assistant reply
  5. Return both records to the frontend
"""

import asyncio
from typing import Annotated, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from ..chat_service import ChatService
from ..core.models import UserRecord
from ..memory.repository import MemoryRepository, MemoryRepositoryError
from .deps import get_chat_service, get_current_user, get_repo
from .schemas import ChatMessageResponse, SendMessageRequest, SendMessageResponse

router = APIRouter(tags=["chat"])


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
def send_message(
    session_id: UUID,
    body: SendMessageRequest,
    _current_user: Annotated[UserRecord, Depends(get_current_user)],
    repo: Annotated[MemoryRepository, Depends(get_repo)],
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
):
    """Persist the user's message, get an LLM response, and return both.

    Uses ChatService.get_response_async internally. FastAPI runs sync endpoints
    in a thread pool, so we use asyncio.run to call the async method from a
    synchronous context without blocking the event loop.
    """
    try:
        # 1. Store user message
        user_record = repo.add_message(
            session_id=session_id,
            sender="user",
            message=body.message,
        )

        # 2. Fetch history (excluding the message we just added — it's already
        #    the last entry, so slice to exclude it for the context prompt)
        history = repo.get_conversation_history(session_id=session_id)
        context_history = history[:-1]  # same pattern as app.py

        # 3. Get LLM response (call async from sync endpoint via asyncio.run)
        assistant_text = asyncio.run(
            chat_service.get_response_async(
                user_message=body.message,
                history=context_history,
            )
        )

        # 4. Store assistant reply
        assistant_record = repo.add_message(
            session_id=session_id,
            sender="assistant",
            message=assistant_text,
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
