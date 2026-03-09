"""Authentication endpoints.

POST /auth/signup  — register a new user
POST /auth/signin  — exchange credentials for a JWT
POST /auth/signout — stateless logout (client discards token)
GET  /auth/me      — return the authenticated user's profile
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..core.models import UserRecord
from ..memory.repository import AuthenticationError, MemoryRepository, MemoryRepositoryError
from .deps import create_access_token, get_current_user, get_repo
from .schemas import SignInRequest, SignUpRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


def _to_user_response(user: UserRecord) -> UserResponse:
    return UserResponse(
        user_id=user.user_id,
        name=user.name,
        email=user.email,
        created_at=user.created_at,
    )


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(
    body: SignUpRequest,
    repo: Annotated[MemoryRepository, Depends(get_repo)],
):
    """Register a new user account."""
    try:
        user = repo.create_user(name=body.name, email=body.email, password=body.password)
    except ValueError as e:
        # create_user raises ValueError if email already exists
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except MemoryRepositoryError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return _to_user_response(user)


@router.post("/signin", response_model=TokenResponse)
def signin(
    body: SignInRequest,
    repo: Annotated[MemoryRepository, Depends(get_repo)],
):
    """Authenticate and return a JWT access token."""
    try:
        user = repo.authenticate_user(email=body.email, password=body.password)
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    except MemoryRepositoryError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    token = create_access_token(str(user.user_id))
    return TokenResponse(access_token=token)


@router.post("/signout", status_code=status.HTTP_204_NO_CONTENT)
def signout(
    _current_user: Annotated[UserRecord, Depends(get_current_user)],
):
    """Stateless logout — the client simply discards the token.

    This endpoint exists so the frontend has a hook for future server-side
    token invalidation (e.g. a blocklist) without any frontend changes.
    """
    return None


@router.get("/me", response_model=UserResponse)
def me(
    current_user: Annotated[UserRecord, Depends(get_current_user)],
):
    """Return the currently authenticated user's profile."""
    return _to_user_response(current_user)
