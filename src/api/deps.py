"""FastAPI dependency injection.

Provides:
- ConfigManager singleton (loaded once at startup)
- ChatService singleton (LLM provider is expensive to initialise)
- JWT token creation and verification
- get_current_user — resolves a Bearer token to a UserRecord
- Singletons (ConfigManager, ChatService) are initialised once in the lifespan
  context manager in main.py and stored on app.state. Per-request dependencies
- (MemoryRepository, get_current_user) are plain functions resolved by FastAPI.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
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
# app.state accessors — called by per-request dependencies
# ---------------------------------------------------------------------------

def get_config(request: Request) -> ConfigManager:
    return request.app.state.config


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


# ---------------------------------------------------------------------------
# Per-request dependencies
# ---------------------------------------------------------------------------

def get_repo(request: Request) -> MemoryRepository:
    """Return a fresh MemoryRepository for each request."""
    return MemoryRepository(request.app.state.config.db_config)


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