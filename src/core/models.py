"""Data models for the application."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class IntersessionConfig:
    """Configuration for the intersession memory background job."""

    enabled: bool = True
    summary_interval_hours: int = 24          # how often to regenerate summaries
    max_summaries_per_prompt: int = 5         # max previous-session summaries to inject
    intersession_context_max_tokens: int = 2000  # token budget for all summaries combined


@dataclass
class ChunkScoringConfig:
    """Configuration for the RLHF chunk-scoring background job."""

    interval_hours: int = 168    # weekly recomputation
    rlhf_alpha: float = 0.2      # weight of quality score vs cosine similarity in retrieval


@dataclass
class JobsConfig:
    """Configuration for all background jobs."""

    intersession: IntersessionConfig = field(default_factory=IntersessionConfig)
    chunk_scoring: ChunkScoringConfig = field(default_factory=ChunkScoringConfig)


@dataclass
class EmbeddingConfig:
    """Configuration for the active embedding provider."""

    provider: str       # "local" | "ollama" | "openai"
    model: str
    dimension: int      # vector dimension — must match what the embedder actually outputs
    api_key: Optional[str] = None
    base_url: Optional[str] = None


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
    short_term_limit: int = 10
    long_term_similarity_threshold: float = 0.70


@dataclass
class GuardrailsConfig:
    """Configuration for input guardrails."""

    enabled: bool = True
    toxicity: bool = True
    bias: bool = True
    prompt_injection: bool = True
    jailbreaking: bool = True
    evaluator_model: str = "gpt-4o-mini"  # model used by DeepEval metrics — use a fast cheap model


@dataclass
class RerankerConfig:
    """Configuration for the cross-encoder reranker."""

    enabled: bool = True
    model: str = "BAAI/bge-reranker-base"
    top_k: int = 5
    device: str = "cpu"


@dataclass
class StorageConfig:
    """Configuration for file storage and distributed state backends."""

    deployment: str               # "local" | "cloud"
    cloud_provider: Optional[str] = None  # "aws" | "azure" | "gcp"
    # AWS-specific
    aws_s3_bucket: Optional[str] = None
    aws_s3_region: Optional[str] = None
    aws_redis_url: Optional[str] = None


@dataclass
class MCPConfig:
    """Configuration for the MCP tools library server connection."""

    enabled: bool = True
    transport: str = "stdio"  # "stdio" | "streamable-http"
    # stdio: launch the MCP server as a local subprocess
    stdio_command: str = "python"
    stdio_args: list = field(default_factory=list)
    stdio_env: dict = field(default_factory=dict)
    # streamable-http: connect to a running remote MCP server
    http_url: str = ""


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
    terminated_at: Optional[datetime] = None


@dataclass
class ChatRecord:
    """Represents a chat message from the database."""

    chat_id: uuid.UUID
    session_id: uuid.UUID
    sender: str
    message: str
    created_at: datetime
    charts: list = field(default_factory=list)