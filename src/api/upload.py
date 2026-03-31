"""File upload endpoint.

POST /sessions/{session_id}/upload — upload a document for a session

The uploaded file is saved to:
    local/active/{user_id}/{session_id}/{filename}

Supported types: PDF, plain-text, PNG, JPG (extensible).
The stored file path is returned so clients can trigger extraction later.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from ..core.models import UserRecord
from ..memory.repository import MemoryRepository, MemoryRepositoryError
from .deps import get_current_user, get_repo
from .loader import FileLoader
from .schemas import UploadResponse

router = APIRouter(prefix="/sessions", tags=["upload"])

_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "image/png",
    "image/jpeg",
}

_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post(
    "/{session_id}/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    session_id: UUID,
    file: UploadFile,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    repo: Annotated[MemoryRepository, Depends(get_repo)],
) -> UploadResponse:
    """Upload a document and store it under the session's active folder.

    - Verifies the session belongs to the authenticated user.
    - Rejects unsupported content types and files that exceed the size limit.
    - Returns the stored file path for use in downstream extraction calls.
    """
    # Verify session ownership
    try:
        session = repo.get_session(session_id=session_id, user_id=current_user.user_id)
    except MemoryRepositoryError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    if not session.is_active:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Session is terminated.")

    # Validate content type
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {sorted(_ALLOWED_CONTENT_TYPES)}",
        )

    content = await file.read()

    if len(content) > _MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {_MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit.",
        )

    loader = FileLoader()
    saved_path = loader.save(
        file_content=content,
        filename=file.filename or "upload",
        user_id=str(current_user.user_id),
        session_id=str(session_id),
    )

    return UploadResponse(
        session_id=session_id,
        filename=file.filename or "upload",
        file_path=str(saved_path),
        size_bytes=len(content),
        content_type=file.content_type or "",
    )
