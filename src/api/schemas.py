"""Pydantic request/response schemas for the REST API.

Field names are kept identical to the frontend TypeScript types so the React
app can consume responses without any transformation layer.
"""

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class SignUpRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class SignUpResponse(BaseModel):
    message: str
    status: str = "pending"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    user_id: UUID
    name: str
    email: str
    created_at: datetime
    is_admin: bool = False

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    session_name: str


class SessionResponse(BaseModel):
    session_id: UUID
    user_id: UUID
    session_name: str
    is_active: bool
    created_at: datetime
    terminated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Chat messages
# ---------------------------------------------------------------------------

class SendMessageRequest(BaseModel):
    message: str
    category: Literal["workflow", "agent"] = "workflow"
    variant: Literal[
        "fast",
        "deep",
        "single_rag_agent",
        "supervisor_orchestration_agent",
    ] = "fast"


class ChatMessageResponse(BaseModel):
    chat_id: UUID
    session_id: UUID
    sender: str
    message: str
    created_at: datetime
    charts: list[str] = []              # base64 PNG charts; populated on live responses only

    model_config = {"from_attributes": True}


class SendMessageResponse(BaseModel):
    """Both the stored user message and the assistant reply in one response."""
    user_message: ChatMessageResponse
    assistant_message: ChatMessageResponse


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: UUID
    chat_id: UUID
    session_id: UUID
    rating: str


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    session_id: UUID
    filename: str
    file_path: str
    size_bytes: int
    content_type: str
    file_description: str = ""
    parent_chunks: int = 0
    child_chunks: int = 0


# ---------------------------------------------------------------------------
# Admin — User management
# ---------------------------------------------------------------------------

class AdminUserResponse(BaseModel):
    user_id: str
    name: Optional[str]
    email: str
    status: str
    created_at: datetime
    last_login_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AdminUserStatusRequest(BaseModel):
    status: Literal["approved", "rejected"]


# ---------------------------------------------------------------------------
# Admin — Conversations
# ---------------------------------------------------------------------------

class AdminSessionResponse(BaseModel):
    session_id: str
    user_email: str
    session_name: Optional[str]
    is_active: bool
    created_at: datetime
    message_count: int
    last_mode: Optional[str] = None

    model_config = {"from_attributes": True}


class AdminMessageResponse(BaseModel):
    chat_id: str
    sender: str
    message: str
    created_at: str
    orchestrator_metadata: dict = {}


# ---------------------------------------------------------------------------
# Admin — Feedback & RLHF
# ---------------------------------------------------------------------------

class AdminFeedbackStats(BaseModel):
    ratings_7d: int
    positive_rate: int
    rlhf_alpha: float


class AdminChunkScore(BaseModel):
    chunk_id: str
    filename: str
    positive_count: int
    negative_count: int
    score: float


# ---------------------------------------------------------------------------
# Admin — Governance
# ---------------------------------------------------------------------------

class AdminGovernanceFlag(BaseModel):
    id: str
    chat_id: str
    session_id: str
    toxicity_score: float
    bias_score: float
    faithfulness_score: Optional[float]
    flagged: bool
    flag_reason: Optional[str]
    created_at: datetime


# ---------------------------------------------------------------------------
# Admin — Background jobs
# ---------------------------------------------------------------------------

class AdminJobStatus(BaseModel):
    job_id: str
    interval_hours: Optional[float]
    next_run: Optional[str]
    last_run: Optional[str]
    status: Optional[str]   # "succeeded" | "failed" | "skipped" | None
    detail: Optional[str]


# ---------------------------------------------------------------------------
# Admin — Knowledge base
# ---------------------------------------------------------------------------

class AdminDocumentResponse(BaseModel):
    filename: str
    file_description: str
    file_type: str
    parent_chunks: int
    child_chunks: int
    ingested_at: Optional[str]


# ---------------------------------------------------------------------------
# Admin — Overview
# ---------------------------------------------------------------------------

class AdminOverviewStats(BaseModel):
    pending_approvals: int
    flagged_responses: int
    active_users_7d: int
    job_failures_24h: int


class AdminActivityEvent(BaseModel):
    event_type: str   # "signup" | "flagged" | "job_run"
    detail: Optional[str]
    occurred_at: str
