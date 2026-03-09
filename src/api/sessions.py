"""Session management endpoints.

GET    /sessions                   — list all sessions for the current user
POST   /sessions                   — create a new session
DELETE /sessions/{session_id}      — hard-delete a session (and its messages)
POST   /sessions/{session_id}/terminate — soft-terminate (mark inactive)
"""

from typing import Annotated, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from ..core.models import UserRecord
from ..memory.repository import MemoryRepository, MemoryRepositoryError
from .deps import get_current_user, get_repo
from .schemas import CreateSessionRequest, SessionResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _to_session_response(s) -> SessionResponse:
    return SessionResponse(
        session_id=s.session_id,
        user_id=s.user_id,
        session_name=s.session_name,
        is_active=s.is_active,
        created_at=s.created_at,
        terminated_at=getattr(s, "terminated_at", None),
    )


@router.get("", response_model=List[SessionResponse])
def list_sessions(
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    repo: Annotated[MemoryRepository, Depends(get_repo)],
):
    """Return all sessions for the authenticated user, newest first."""
    try:
        sessions = repo.get_sessions(current_user.user_id)
    except MemoryRepositoryError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return [_to_session_response(s) for s in sessions]


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    body: CreateSessionRequest,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    repo: Annotated[MemoryRepository, Depends(get_repo)],
):
    """Create a new chat session for the authenticated user."""
    try:
        session = repo.create_session(
            user_id=current_user.user_id,
            session_name=body.session_name,
        )
    except MemoryRepositoryError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return _to_session_response(session)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: UUID,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    repo: Annotated[MemoryRepository, Depends(get_repo)],
):
    """Hard-delete a session and all its messages (CASCADE).

    Used by the frontend to clean up empty sessions created but never used.
    """
    try:
        repo.delete_session(session_id=session_id, user_id=current_user.user_id)
    except MemoryRepositoryError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return None


@router.post("/{session_id}/terminate", status_code=status.HTTP_204_NO_CONTENT)
def terminate_session(
    session_id: UUID,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    repo: Annotated[MemoryRepository, Depends(get_repo)],
):
    """Soft-terminate a session (sets is_active=False, stamps terminated_at).

    The session and its messages are preserved — use DELETE to remove entirely.
    """
    try:
        repo.terminate_session(session_id=session_id)
    except MemoryRepositoryError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return None
