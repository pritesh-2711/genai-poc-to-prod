"""API tests for /sessions endpoints.

Uses the sessions_client fixture from conftest.py where both get_repo and
get_current_user are overridden so tests focus on session CRUD behaviour.
"""

import pytest
from datetime import datetime
from uuid import UUID

from src.core.models import SessionRecord
from src.memory.repository import MemoryRepositoryError

TEST_USER_ID = UUID("00000000-0000-0000-0000-000000000001")
TEST_SESSION_ID = UUID("00000000-0000-0000-0000-000000000002")
OTHER_SESSION_ID = UUID("00000000-0000-0000-0000-000000000003")


def _session(session_id=TEST_SESSION_ID, is_active=True, name="My Session") -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        user_id=TEST_USER_ID,
        session_name=name,
        is_active=is_active,
        created_at=datetime(2024, 1, 1, 12, 0, 0),
    )


# ---------------------------------------------------------------------------
# POST /sessions
# ---------------------------------------------------------------------------

def test_create_session_returns_201(sessions_client, mock_repo) -> None:
    mock_repo.create_session.return_value = _session()
    resp = sessions_client.post("/sessions", json={"session_name": "My Session"})
    assert resp.status_code == 201


def test_create_session_returns_session_object(sessions_client, mock_repo) -> None:
    mock_repo.create_session.return_value = _session(name="Research Chat")
    resp = sessions_client.post("/sessions", json={"session_name": "Research Chat"})
    data = resp.json()
    assert data["session_name"] == "Research Chat"
    assert data["is_active"] is True
    assert "session_id" in data


def test_create_session_db_error_returns_500(sessions_client, mock_repo) -> None:
    mock_repo.create_session.side_effect = MemoryRepositoryError("db error")
    resp = sessions_client.post("/sessions", json={"session_name": "Fail"})
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# GET /sessions
# ---------------------------------------------------------------------------

def test_list_sessions_returns_200(sessions_client, mock_repo) -> None:
    mock_repo.get_sessions.return_value = [_session()]
    resp = sessions_client.get("/sessions")
    assert resp.status_code == 200


def test_list_sessions_returns_only_user_sessions(sessions_client, mock_repo) -> None:
    sessions = [_session(session_id=TEST_SESSION_ID, name="S1"),
                _session(session_id=OTHER_SESSION_ID, name="S2")]
    mock_repo.get_sessions.return_value = sessions
    resp = sessions_client.get("/sessions")
    data = resp.json()
    assert len(data) == 2
    # Verify get_sessions was called with the test user's ID
    mock_repo.get_sessions.assert_called_once_with(TEST_USER_ID)


def test_list_sessions_empty_returns_empty_list(sessions_client, mock_repo) -> None:
    mock_repo.get_sessions.return_value = []
    resp = sessions_client.get("/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_sessions_db_error_returns_500(sessions_client, mock_repo) -> None:
    mock_repo.get_sessions.side_effect = MemoryRepositoryError("db error")
    resp = sessions_client.get("/sessions")
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# DELETE /sessions/{session_id}
# ---------------------------------------------------------------------------

def test_delete_session_returns_204(sessions_client, mock_repo) -> None:
    mock_repo.delete_session.return_value = None
    resp = sessions_client.delete(f"/sessions/{TEST_SESSION_ID}")
    assert resp.status_code == 204


def test_delete_session_calls_repo_with_correct_ids(sessions_client, mock_repo) -> None:
    mock_repo.delete_session.return_value = None
    sessions_client.delete(f"/sessions/{TEST_SESSION_ID}")
    mock_repo.delete_session.assert_called_once_with(
        session_id=TEST_SESSION_ID,
        user_id=TEST_USER_ID,
    )


def test_delete_session_calls_file_loader_archive(sessions_client, mock_repo) -> None:
    mock_repo.delete_session.return_value = None
    sessions_client.delete(f"/sessions/{TEST_SESSION_ID}")
    # The route calls app.state.file_loader.archive() after repo delete
    file_loader = sessions_client.app.state.file_loader
    file_loader.archive.assert_called_once()


def test_delete_session_db_error_returns_500(sessions_client, mock_repo) -> None:
    mock_repo.delete_session.side_effect = MemoryRepositoryError("db error")
    resp = sessions_client.delete(f"/sessions/{TEST_SESSION_ID}")
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/terminate
# ---------------------------------------------------------------------------

def test_terminate_session_returns_204(sessions_client, mock_repo) -> None:
    mock_repo.terminate_session.return_value = None
    resp = sessions_client.post(f"/sessions/{TEST_SESSION_ID}/terminate")
    assert resp.status_code == 204


def test_terminate_session_sets_is_active_false(sessions_client, mock_repo) -> None:
    mock_repo.terminate_session.return_value = None
    sessions_client.post(f"/sessions/{TEST_SESSION_ID}/terminate")
    mock_repo.terminate_session.assert_called_once_with(session_id=TEST_SESSION_ID)


def test_terminate_session_db_error_returns_500(sessions_client, mock_repo) -> None:
    mock_repo.terminate_session.side_effect = MemoryRepositoryError("db error")
    resp = sessions_client.post(f"/sessions/{TEST_SESSION_ID}/terminate")
    assert resp.status_code == 500
