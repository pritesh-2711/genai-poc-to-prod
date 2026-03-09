"""FastAPI dependency injection.

Provides:
- ConfigManager singleton (loaded once at startup)
- MemoryRepository factory (lightweight — new instance per request)
- ChatService singleton (LLM provider is expensive to initialise)
- JWT token creation and verification
- get_current_user — resolves a Bearer token to a UserRecord
"""

import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from jose import JWTError, jwt

from ..chat_service import ChatService
from ..core.config import ConfigManager
from ..core.models import UserRecord
from ..memory.repository import AuthenticationError, MemoryRepository, MemoryRepositoryError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

bearer_scheme = HTTPBearer()


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_config() -> ConfigManager:
    """Load and cache the ConfigManager once per process."""
    return ConfigManager()


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """Create and cache the ChatService (LLM provider) once per process."""
    cfg = get_config()
    return ChatService(llm_config=cfg.llm_config, chat_config=cfg.chat_config)


# ---------------------------------------------------------------------------
# Per-request dependencies
# ---------------------------------------------------------------------------

def get_repo() -> MemoryRepository:
    """Return a fresh MemoryRepository for each request.

    MemoryRepository opens and closes its own psycopg2 connection per call,
    so creating a new instance per request is cheap and keeps things simple
    (consistent with the existing Chainlit usage pattern).
    """
    cfg = get_config()
    return MemoryRepository(cfg.db_config)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _get_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "JWT_SECRET_KEY environment variable is not set. "
            "Add it to your .env file."
        )
    return secret


def create_access_token(user_id: str) -> str:
    """Encode a JWT that expires in ACCESS_TOKEN_EXPIRE_DAYS days."""
    expire = datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)


def _decode_token(token: str) -> str:
    """Decode token and return the user_id (sub claim).

    Raises HTTPException 401 on any failure.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, _get_secret(), algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return user_id
    except JWTError:
        raise credentials_exception


# ---------------------------------------------------------------------------
# Current-user dependency
# ---------------------------------------------------------------------------

def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
    repo: Annotated[MemoryRepository, Depends(get_repo)],
) -> UserRecord:
    """Verify the Bearer token and return the corresponding UserRecord."""
    user_id = _decode_token(credentials.credentials)
    try:
        user = repo.get_user_by_id(user_id)
    except MemoryRepositoryError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
