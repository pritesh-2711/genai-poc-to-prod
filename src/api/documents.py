"""Document listing endpoint.

GET /sessions/{session_id}/documents — list ingested documents for a session.

Returns one entry per unique filename ingested into the session, with
aggregate chunk counts so the UI can display ingestion stats.
"""

from typing import Annotated
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from ..core.models import UserRecord
from ..memory.repository import MemoryRepository, MemoryRepositoryError
from .deps import get_current_user, get_repo

router = APIRouter(prefix="/sessions", tags=["documents"])


class DocumentRecord(BaseModel):
    filename: str
    file_description: str
    file_type: str
    parent_chunks: int
    child_chunks: int
    ingested_at: str  # ISO timestamp of the first ingestion for this file


async def _fetch_session_documents(
    db_config,
    session_id: UUID,
) -> list[DocumentRecord]:
    """Query poc2prod.ingestions grouped by filename for a session."""
    conn = await asyncpg.connect(
        host=db_config.host,
        port=db_config.port,
        database=db_config.database,
        user=db_config.user,
        password=db_config.password,
        server_settings={"search_path": "poc2prod,public"},
    )
    try:
        rows = await conn.fetch(
            """
            SELECT
                filename,
                COALESCE(MAX(file_description), '')  AS file_description,
                COALESCE(MAX(type), 'pdf')            AS file_type,
                COUNT(DISTINCT parent_id)             AS parent_chunks,
                COUNT(id)                             AS child_chunks,
                MIN(created_at)::text                 AS ingested_at
            FROM poc2prod.ingestions
            WHERE session_id = $1
            GROUP BY filename
            ORDER BY MIN(created_at) DESC;
            """,
            str(session_id),
        )
    finally:
        await conn.close()

    return [
        DocumentRecord(
            filename=row["filename"],
            file_description=row["file_description"],
            file_type=row["file_type"],
            parent_chunks=row["parent_chunks"],
            child_chunks=row["child_chunks"],
            ingested_at=row["ingested_at"],
        )
        for row in rows
    ]


@router.get(
    "/{session_id}/documents",
    response_model=list[DocumentRecord],
)
async def list_session_documents(
    session_id: UUID,
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    repo: Annotated[MemoryRepository, Depends(get_repo)],
) -> list[DocumentRecord]:
    """Return all documents ingested for the given session.

    Verifies session ownership before querying. Returns one record per unique
    filename with aggregate parent and child chunk counts.
    """
    try:
        session = repo.get_session(session_id=session_id, user_id=current_user.user_id)
    except MemoryRepositoryError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    db_config = request.app.state.config.db_config
    return await _fetch_session_documents(db_config, session_id)
