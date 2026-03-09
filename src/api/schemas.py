"""Pydantic request/response schemas for the REST API.

Field names are kept identical to the frontend TypeScript types so the React
app can consume responses without any transformation layer.
"""

from datetime import datetime
from typing import Optional
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


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    user_id: UUID
    name: str
    email: str
    created_at: datetime

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


class ChatMessageResponse(BaseModel):
    chat_id: UUID
    session_id: UUID
    sender: str
    message: str
    created_at: datetime

    model_config = {"from_attributes": True}


class SendMessageResponse(BaseModel):
    """Both the stored user message and the assistant reply in one response."""
    user_message: ChatMessageResponse
    assistant_message: ChatMessageResponse
