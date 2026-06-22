"""Shared test fixtures for the ch-15 backend test suite.

Layers
------
unit        : pure Python, no DB, no LLM
api         : FastAPI TestClient with mocked repositories
integration : real test DB (marked with @pytest.mark.integration)
evaluation  : live LLM (opt-in with --run-eval flag)
"""

import os

# Must be set before any app module is imported.
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")

import pytest
from datetime import datetime
from unittest.mock import MagicMock
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.auth import router as auth_router
from src.api.sessions import router as sessions_router
from src.api.deps import get_repo, get_current_user, create_access_token
from src.core.models import DBConfig, SessionRecord, UserRecord

# ---------------------------------------------------------------------------
# Fixed test identifiers
# ---------------------------------------------------------------------------

TEST_USER_ID: UUID = UUID("00000000-0000-0000-0000-000000000001")
TEST_SESSION_ID: UUID = UUID("00000000-0000-0000-0000-000000000002")


# ---------------------------------------------------------------------------
# Core data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_user() -> UserRecord:
    return UserRecord(
        user_id=TEST_USER_ID,
        name="Test User",
        email="test@example.com",
        created_at=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def test_session(test_user: UserRecord) -> SessionRecord:
    return SessionRecord(
        session_id=TEST_SESSION_ID,
        user_id=test_user.user_id,
        session_name="Test Session",
        is_active=True,
        created_at=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def mock_repo() -> MagicMock:
    """A fresh MagicMock standing in for MemoryRepository."""
    return MagicMock()


@pytest.fixture
def auth_token(test_user: UserRecord) -> str:
    """Valid JWT for test_user, signed with the test secret."""
    return create_access_token(str(test_user.user_id))


# ---------------------------------------------------------------------------
# API test clients
# ---------------------------------------------------------------------------

@pytest.fixture
def auth_client(mock_repo: MagicMock) -> TestClient:
    """Auth router test client.

    Only ``get_repo`` is overridden so that JWT verification still runs.
    Tests for expired/missing tokens work correctly with this client.
    """
    app = FastAPI()
    app.include_router(auth_router)
    app.dependency_overrides[get_repo] = lambda: mock_repo
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sessions_client(mock_repo: MagicMock, test_user: UserRecord) -> TestClient:
    """Sessions router test client.

    Both ``get_repo`` and ``get_current_user`` are overridden so that session
    CRUD tests are not concerned with JWT mechanics.
    """
    app = FastAPI()
    app.include_router(sessions_router)
    app.dependency_overrides[get_repo] = lambda: mock_repo
    app.dependency_overrides[get_current_user] = lambda: test_user
    # delete_session accesses app.state.file_loader
    app.state.file_loader = MagicMock()
    return TestClient(app, raise_server_exceptions=False)
