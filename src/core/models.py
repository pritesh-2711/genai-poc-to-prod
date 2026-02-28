"""Data models for the application."""

from dataclasses import dataclass
from typing import Optional


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
    max_history: int = 10
    timeout: int = 30
