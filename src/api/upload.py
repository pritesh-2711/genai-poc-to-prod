"""File upload endpoint.

POST /sessions/{session_id}/upload — upload a document and ingest it.

Flow:
    1. Validate session ownership and file type (PDF or DOCX only).
    2. Save the file to storage/{user_id}/active/{session_id}/{filename}.
    3. Run the IngestionPipeline: extract → chunk → embed → INSERT into DB.
    4. Return the saved path, file stats, and ingestion chunk counts.

Storage layout:
    storage/{user_id}/active/{session_id}/{filename}   ← active session
    storage/{user_id}/archive/{session_id}/{filename}  ← after session deleted
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, status

from ..core.models import UserRecord
from ..databases.pipeline import IngestionPipeline, IngestionPipelineError
from ..memory.repository import MemoryRepository, MemoryRepositoryError
from .deps import get_current_user, get_ingestion_pipeline, get_repo
from .loader import BaseFileLoader
from .schemas import UploadResponse

router = APIRouter(prefix="/sessions", tags=["upload"])

_ALLOWED_CONTENT_TYPES: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "doc",
}

_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post(
    "/{session_id}/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    session_id: UUID,
    request: Request,
    file: UploadFile,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    repo: Annotated[MemoryRepository, Depends(get_repo)],
    pipeline: Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)],
    file_description: str = Form(""),
) -> UploadResponse:
    """Upload a document, extract its content, and ingest it into the vector store.

    - Verifies the session belongs to the authenticated user.
    - Accepts PDF and DOCX files up to 50 MB.
    - Runs extraction → hierarchical chunking → embedding → DB insertion
      synchronously and returns chunk counts on completion.
    """
    # ------------------------------------------------------------------
    # Verify session ownership
    # ------------------------------------------------------------------
    try:
        session = repo.get_session(session_id=session_id, user_id=current_user.user_id)
    except MemoryRepositoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found."
        )
    if not session.is_active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Session is terminated."
        )

    # ------------------------------------------------------------------
    # Validate content type
    # ------------------------------------------------------------------
    content_type = file.content_type or ""
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{content_type}'. "
                f"Allowed: PDF, DOCX."
            ),
        )

    file_type = _ALLOWED_CONTENT_TYPES[content_type]

    # ------------------------------------------------------------------
    # Read + size check
    # ------------------------------------------------------------------
    content = await file.read()
    if len(content) > _MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {_MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit.",
        )

    filename = file.filename or "upload"

    # ------------------------------------------------------------------
    # Save to active storage (local path or S3 + tempfile)
    # ------------------------------------------------------------------
    loader: BaseFileLoader = request.app.state.file_loader
    saved_path = loader.save(
        file_content=content,
        filename=filename,
        user_id=str(current_user.user_id),
        session_id=str(session_id),
    )

    # ------------------------------------------------------------------
    # Extract → chunk → embed → ingest
    # cleanup_temp() removes the tempfile for S3; no-op for local.
    # ------------------------------------------------------------------
    try:
        result = await pipeline.run(
            file_path=saved_path,
            user_id=current_user.user_id,
            session_id=session_id,
            file_description=file_description,
            file_type=file_type,
        )
    except IngestionPipelineError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Ingestion failed: {exc}",
        )
    finally:
        loader.cleanup_temp(saved_path)

    return UploadResponse(
        session_id=session_id,
        filename=filename,
        file_path=str(saved_path),
        size_bytes=len(content),
        content_type=content_type,
        file_description=file_description,
        parent_chunks=result.parent_count,
        child_chunks=result.child_count,
    )
