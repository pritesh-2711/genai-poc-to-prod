"""API tests for /auth endpoints.

Uses the auth_client fixture from conftest.py which overrides only get_repo,
so JWT verification still runs for tests that exercise GET /auth/me.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import UUID

from jose import jwt

from src.api.deps import create_access_token, ALGORITHM, ACCESS_TOKEN_EXPIRE_DAYS
from src.core.models import UserRecord
from src.memory.repository import (
    AuthenticationError,
    MemoryRepositoryError,
    UserNotApprovedError,
)

TEST_USER_ID = UUID("00000000-0000-0000-0000-000000000001")

_USER = UserRecord(
    user_id=TEST_USER_ID,
    name="Jane Doe",
    email="jane@example.com",
    created_at=datetime(2024, 1, 1),
)


# ---------------------------------------------------------------------------
# POST /auth/signup
# ---------------------------------------------------------------------------

def test_signup_valid_payload_returns_201(auth_client, mock_repo) -> None:
    mock_repo.create_user.return_value = _USER
    resp = auth_client.post(
        "/auth/signup",
        json={"name": "Jane Doe", "email": "jane@example.com", "password": "secret123"},
    )
    assert resp.status_code == 201
    assert resp.json()["status"] == "pending"


def test_signup_duplicate_email_returns_409(auth_client, mock_repo) -> None:
    mock_repo.create_user.side_effect = ValueError("already exists")
    resp = auth_client.post(
        "/auth/signup",
        json={"name": "Jane Doe", "email": "jane@example.com", "password": "secret123"},
    )
    assert resp.status_code == 409


def test_signup_db_error_returns_500(auth_client, mock_repo) -> None:
    mock_repo.create_user.side_effect = MemoryRepositoryError("db down")
    resp = auth_client.post(
        "/auth/signup",
        json={"name": "Jane Doe", "email": "jane@example.com", "password": "secret123"},
    )
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /auth/signin
# ---------------------------------------------------------------------------

def test_signin_approved_account_returns_token(auth_client, mock_repo) -> None:
    mock_repo.authenticate_user.return_value = _USER
    resp = auth_client.post(
        "/auth/signin",
        json={"email": "jane@example.com", "password": "secret123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_signin_wrong_password_returns_401(auth_client, mock_repo) -> None:
    mock_repo.authenticate_user.side_effect = AuthenticationError("bad credentials")
    resp = auth_client.post(
        "/auth/signin",
        json={"email": "jane@example.com", "password": "wrongpass"},
    )
    assert resp.status_code == 401


def test_signin_pending_account_returns_403(auth_client, mock_repo) -> None:
    mock_repo.authenticate_user.side_effect = UserNotApprovedError("pending")
    resp = auth_client.post(
        "/auth/signin",
        json={"email": "jane@example.com", "password": "secret123"},
    )
    assert resp.status_code == 403


def test_signin_declined_account_returns_403(auth_client, mock_repo) -> None:
    mock_repo.authenticate_user.side_effect = UserNotApprovedError("declined")
    resp = auth_client.post(
        "/auth/signin",
        json={"email": "jane@example.com", "password": "secret123"},
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------

def test_me_with_valid_token_returns_user_profile(auth_client, mock_repo, auth_token) -> None:
    mock_repo.get_user_by_id.return_value = _USER
    resp = auth_client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == "jane@example.com"
    assert data["name"] == "Jane Doe"
    assert "user_id" in data


def test_me_with_no_token_returns_401(auth_client) -> None:
    resp = auth_client.get("/auth/me")
    # HTTPBearer returns 401 when no Authorization header is present
    assert resp.status_code == 401


def test_me_with_expired_token_returns_401(auth_client) -> None:
    import os
    secret = os.environ["JWT_SECRET_KEY"]
    expired_payload = {
        "sub": str(TEST_USER_ID),
        "exp": datetime.now(timezone.utc) - timedelta(days=1),
    }
    expired_token = jwt.encode(expired_payload, secret, algorithm=ALGORITHM)
    resp = auth_client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert resp.status_code == 401


def test_me_with_invalid_token_returns_401(auth_client) -> None:
    resp = auth_client.get(
        "/auth/me",
        headers={"Authorization": "Bearer not.a.valid.jwt"},
    )
    assert resp.status_code == 401


def test_me_with_unknown_user_id_returns_401(auth_client, mock_repo, auth_token) -> None:
    mock_repo.get_user_by_id.return_value = None
    resp = auth_client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 401
