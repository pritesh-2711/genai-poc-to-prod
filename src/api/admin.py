"""Admin API endpoints.

All routes require the caller's email to be listed in the ADMIN_EMAILS
environment variable (checked by the `require_admin` dependency).

Prefix: /admin
Tags:   ["admin"]

Sections
--------
GET  /admin/overview                          Summary stats + recent activity
GET  /admin/users                             All users (optional ?search=)
GET  /admin/users/pending                     Users awaiting approval
POST /admin/users/{user_id}/approve           Approve a pending user
POST /admin/users/{user_id}/reject            Reject a pending user

GET  /admin/conversations                     All sessions (optional ?search=)
GET  /admin/conversations/{session_id}        Chat messages in a session
GET  /admin/conversations/summaries           Session summaries (intersession memory)

GET  /admin/feedback                          Aggregate feedback stats + chunk scores

GET  /admin/governance                        Output-guardrail flags (all or ?flagged_only=true)

GET  /admin/jobs                              Background job statuses

GET  /admin/knowledge-base                    All ingested documents
DELETE /admin/knowledge-base/{filename}       Delete all chunks for a document
"""

import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from ..databases.admin import AdminRepository
from .deps import get_job_history, get_scheduler, require_admin
from .schemas import (
    AdminActivityEvent,
    AdminChunkScore,
    AdminDocumentResponse,
    AdminFeedbackStats,
    AdminGovernanceFlag,
    AdminJobStatus,
    AdminMessageResponse,
    AdminOverviewStats,
    AdminSessionResponse,
    AdminUserResponse,
    AdminUserStatusRequest,
)

router = APIRouter(prefix="/admin", tags=["admin"])


def _get_admin_repo(request: Request) -> AdminRepository:
    return request.app.state.admin_repo


# ── Overview ──────────────────────────────────────────────────────────────────

@router.get("/overview", response_model=dict)
async def overview(
    _admin=Depends(require_admin),
    admin_repo: AdminRepository = Depends(_get_admin_repo),
    job_history: dict = Depends(get_job_history),
):
    stats = await admin_repo.get_overview_stats()
    activity = await admin_repo.get_recent_activity(limit=10)

    # Count job failures in the last 24 h from the in-memory history
    job_failures = sum(
        1 for v in job_history.values()
        if v.get("status") == "failed"
    )

    return {
        "stats": AdminOverviewStats(
            pending_approvals=stats["pending_approvals"],
            flagged_responses=stats["flagged_responses"],
            active_users_7d=stats["active_users_7d"],
            job_failures_24h=job_failures,
        ).model_dump(),
        "recent_activity": [
            AdminActivityEvent(**e).model_dump() for e in activity
        ],
    }


# ── User management ───────────────────────────────────────────────────────────

@router.get("/users", response_model=list[AdminUserResponse])
async def list_users(
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _admin=Depends(require_admin),
    admin_repo: AdminRepository = Depends(_get_admin_repo),
):
    rows = await admin_repo.get_all_users(search=search, limit=limit, offset=offset)
    return [AdminUserResponse(**r) for r in rows]


@router.get("/users/pending", response_model=list[AdminUserResponse])
async def list_pending_users(
    _admin=Depends(require_admin),
    admin_repo: AdminRepository = Depends(_get_admin_repo),
):
    rows = await admin_repo.get_pending_users()
    return [AdminUserResponse(**r) for r in rows]


@router.post("/users/{user_id}/approve", status_code=status.HTTP_204_NO_CONTENT)
async def approve_user(
    user_id: uuid.UUID,
    _admin=Depends(require_admin),
    admin_repo: AdminRepository = Depends(_get_admin_repo),
):
    updated = await admin_repo.set_user_status(user_id, "approved")
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")


@router.post("/users/{user_id}/reject", status_code=status.HTTP_204_NO_CONTENT)
async def reject_user(
    user_id: uuid.UUID,
    _admin=Depends(require_admin),
    admin_repo: AdminRepository = Depends(_get_admin_repo),
):
    updated = await admin_repo.set_user_status(user_id, "rejected")
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")


# ── Conversations ─────────────────────────────────────────────────────────────

@router.get("/conversations", response_model=list[AdminSessionResponse])
async def list_sessions(
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _admin=Depends(require_admin),
    admin_repo: AdminRepository = Depends(_get_admin_repo),
):
    rows = await admin_repo.get_all_sessions(search=search, limit=limit, offset=offset)
    return [AdminSessionResponse(**r) for r in rows]


@router.get("/conversations/summaries")
async def list_session_summaries(
    limit: int = Query(20, ge=1, le=100),
    _admin=Depends(require_admin),
    admin_repo: AdminRepository = Depends(_get_admin_repo),
):
    return await admin_repo.get_session_summaries(limit=limit)


@router.get("/conversations/{session_id}", response_model=list[AdminMessageResponse])
async def get_session_messages(
    session_id: uuid.UUID,
    _admin=Depends(require_admin),
    admin_repo: AdminRepository = Depends(_get_admin_repo),
):
    return await admin_repo.get_session_messages(session_id)


# ── Feedback & RLHF ──────────────────────────────────────────────────────────

@router.get("/feedback")
async def get_feedback(
    request: Request,
    _admin=Depends(require_admin),
    admin_repo: AdminRepository = Depends(_get_admin_repo),
):
    stats = await admin_repo.get_feedback_stats()
    chunks = await admin_repo.get_chunk_scores(limit=50)

    # rlhf_alpha lives in the jobs config
    rlhf_alpha = request.app.state.config.jobs_config.chunk_scoring.rlhf_alpha

    return {
        "stats": AdminFeedbackStats(
            ratings_7d=stats["ratings_7d"],
            positive_rate=stats["positive_rate"],
            rlhf_alpha=rlhf_alpha,
        ).model_dump(),
        "chunk_scores": [AdminChunkScore(**c).model_dump() for c in chunks],
    }


# ── Governance ────────────────────────────────────────────────────────────────

@router.get("/governance", response_model=list[AdminGovernanceFlag])
async def get_governance(
    flagged_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    _admin=Depends(require_admin),
    admin_repo: AdminRepository = Depends(_get_admin_repo),
):
    rows = await admin_repo.get_governance_flags(only_flagged=flagged_only, limit=limit)
    return [AdminGovernanceFlag(**r) for r in rows]


# ── Background jobs ───────────────────────────────────────────────────────────

@router.get("/jobs", response_model=list[AdminJobStatus])
def get_jobs(
    _admin=Depends(require_admin),
    scheduler=Depends(get_scheduler),
    job_history: dict = Depends(get_job_history),
):
    statuses = []
    for job in scheduler.get_jobs():
        history = job_history.get(job.id, {})
        next_run = job.next_run_time.isoformat() if job.next_run_time else None

        # Extract interval from trigger if available
        interval_hours = None
        trigger = job.trigger
        if hasattr(trigger, "interval"):
            total_seconds = trigger.interval.total_seconds()
            interval_hours = total_seconds / 3600

        statuses.append(AdminJobStatus(
            job_id=job.id,
            interval_hours=interval_hours,
            next_run=next_run,
            last_run=history.get("last_run"),
            status=history.get("status"),
            detail=history.get("detail"),
        ))
    return statuses


# ── Knowledge base ────────────────────────────────────────────────────────────

@router.get("/knowledge-base", response_model=list[AdminDocumentResponse])
async def list_documents(
    _admin=Depends(require_admin),
    admin_repo: AdminRepository = Depends(_get_admin_repo),
):
    rows = await admin_repo.get_all_documents()
    return [AdminDocumentResponse(**r) for r in rows]


@router.delete("/knowledge-base/{filename}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    filename: str,
    _admin=Depends(require_admin),
    admin_repo: AdminRepository = Depends(_get_admin_repo),
):
    deleted = await admin_repo.delete_document(filename)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for '{filename}'.")
