"""Data models for the application."""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import uuid


@dataclass
class ChatMessage:
    """Represents a chat message."""

    role: str  # "user" or "assistant"
    content: str
    metadata: Optional[dict] = None


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""

    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class ChatConfig:
    """Configuration for chat service."""

    system_prompt: str
    timeout: int = 60


@dataclass
class DBConfig:
    """Configuration for PostgreSQL database."""

    host: str
    port: int
    database: str
    user: str
    password: str


@dataclass
class UserRecord:
    """Represents a user from the database."""

    user_id: uuid.UUID
    name: str
    email: str
    created_at: datetime


@dataclass
class SessionRecord:
    """Represents a session from the database."""

    session_id: uuid.UUID
    user_id: uuid.UUID
    session_name: str
    is_active: bool
    created_at: datetime


@dataclass
class ChatRecord:
    """Represents a chat message from the database."""

    chat_id: uuid.UUID
    session_id: uuid.UUID
    sender: str
    message: str
    created_at: datetime