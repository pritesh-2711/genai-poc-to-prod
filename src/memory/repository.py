"""PostgreSQL-backed repository for users, sessions, and chat history."""

from typing import List, Optional
import uuid

import bcrypt
import psycopg2
from psycopg2.extras import RealDictCursor

from ..core.exceptions import ResearchPaperChatException
from ..core.logging import LoggingManager
from ..core.models import ChatRecord, DBConfig, SessionRecord, UserRecord

logger = LoggingManager.get_logger(__name__)


class MemoryRepositoryError(ResearchPaperChatException):
    """Raised when a memory repository operation fails."""
    pass


class AuthenticationError(ResearchPaperChatException):
    """Raised when user authentication fails."""
    pass


class MemoryRepository:
    """Handles all database interactions for users, sessions, and chat history.

    Each method opens and closes its own connection via try/finally to prevent
    connection leaks on exceptions.
    """

    def __init__(self, db_config: DBConfig):
        self.db_config = db_config

    def _connect(self):
        return psycopg2.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            database=self.db_config.database,
            user=self.db_config.user,
            password=self.db_config.password,
        )

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def create_user(self, name: str, email: str, password: str) -> UserRecord:
        """Hash the password and insert a new user row.

        Raises:
            ValueError: If the email address is already registered.
            MemoryRepositoryError: On any database error.
        """
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        conn = self._connect()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            try:
                cur.execute(
                    """
                    INSERT INTO memory.users (name, email, password)
                    VALUES (%s, %s, %s)
                    RETURNING user_id, name, email, created_at;
                    """,
                    (name, email, hashed),
                )
                row = cur.fetchone()
                conn.commit()
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                raise ValueError(f"An account with email '{email}' already exists.")
            except Exception as e:
                conn.rollback()
                logger.error(f"DB error creating user: {e}")
                raise MemoryRepositoryError(f"Database error: {e}")
            finally:
                cur.close()
        finally:
            conn.close()

        logger.info(f"User created: {email}")
        return UserRecord(
            user_id=row["user_id"],
            name=row["name"],
            email=row["email"],
            created_at=row["created_at"],
        )

    def get_user_by_id(self, user_id: uuid.UUID) -> Optional[UserRecord]:
        """Return a UserRecord by primary key, or None if not found."""
        conn = self._connect()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            try:
                cur.execute(
                    "SELECT user_id, name, email, created_at FROM memory.users WHERE user_id = %s;",
                    (str(user_id),),
                )
                row = cur.fetchone()
            except Exception as e:
                logger.error(f"DB error fetching user by id: {e}")
                raise MemoryRepositoryError(f"Database error: {e}")
            finally:
                cur.close()
        finally:
            conn.close()

        if row is None:
            return None
        return UserRecord(
            user_id=row["user_id"],
            name=row["name"],
            email=row["email"],
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate_user(self, email: str, password: str) -> UserRecord:
        """Verify credentials, update last_login_at, and return the user record.

        Raises:
            AuthenticationError: If credentials are invalid.
            MemoryRepositoryError: On database error.
        """
        conn = self._connect()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            try:
                cur.execute(
                    "SELECT user_id, name, email, password, created_at FROM memory.users WHERE email = %s;",
                    (email,),
                )
                row = cur.fetchone()
            except Exception as e:
                logger.error(f"DB error during authentication: {e}")
                raise MemoryRepositoryError(f"Database error: {e}")

            if row is None:
                raise AuthenticationError("Invalid email or password.")

            if not bcrypt.checkpw(password.encode("utf-8"), row["password"].encode("utf-8")):
                raise AuthenticationError("Invalid email or password.")

            try:
                cur.execute(
                    "UPDATE memory.users SET last_login_at = CURRENT_TIMESTAMP WHERE user_id = %s;",
                    (str(row["user_id"]),),
                )
                conn.commit()
            except Exception as e:
                logger.warning(f"Failed to update last_login_at for {email}: {e}")
            finally:
                cur.close()
        finally:
            conn.close()

        logger.info(f"User authenticated: {email}")
        return UserRecord(
            user_id=row["user_id"],
            name=row["name"],
            email=row["email"],
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def get_sessions(self, user_id: uuid.UUID) -> List[SessionRecord]:
        """Return all sessions for a user, newest first."""
        conn = self._connect()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            try:
                cur.execute(
                    """
                    SELECT session_id, user_id, session_name, is_active, created_at, terminated_at
                    FROM memory.sessions
                    WHERE user_id = %s
                    ORDER BY created_at DESC;
                    """,
                    (str(user_id),),
                )
                rows = cur.fetchall()
            except Exception as e:
                logger.error(f"DB error fetching sessions: {e}")
                raise MemoryRepositoryError(f"Database error: {e}")
            finally:
                cur.close()
        finally:
            conn.close()

        return [
            SessionRecord(
                session_id=row["session_id"],
                user_id=row["user_id"],
                session_name=row["session_name"],
                is_active=row["is_active"],
                created_at=row["created_at"],
                terminated_at=row["terminated_at"],
            )
            for row in rows
        ]

    def create_session(self, user_id: uuid.UUID, session_name: str) -> SessionRecord:
        """Create a new session for a user."""
        conn = self._connect()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            try:
                cur.execute(
                    """
                    INSERT INTO memory.sessions (user_id, session_name)
                    VALUES (%s, %s)
                    RETURNING session_id, user_id, session_name, is_active, created_at, terminated_at;
                    """,
                    (str(user_id), session_name),
                )
                row = cur.fetchone()
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"DB error creating session: {e}")
                raise MemoryRepositoryError(f"Database error: {e}")
            finally:
                cur.close()
        finally:
            conn.close()

        logger.info(f"Session created: {row['session_id']} for user {user_id}")
        return SessionRecord(
            session_id=row["session_id"],
            user_id=row["user_id"],
            session_name=row["session_name"],
            is_active=row["is_active"],
            created_at=row["created_at"],
            terminated_at=row["terminated_at"],
        )

    def terminate_session(self, session_id: uuid.UUID) -> None:
        """Mark a session as inactive and stamp terminated_at."""
        conn = self._connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    UPDATE memory.sessions
                    SET is_active = FALSE, terminated_at = CURRENT_TIMESTAMP
                    WHERE session_id = %s;
                    """,
                    (str(session_id),),
                )
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"DB error terminating session: {e}")
                raise MemoryRepositoryError(f"Database error: {e}")
            finally:
                cur.close()
        finally:
            conn.close()

    def delete_session(self, session_id: uuid.UUID, user_id: uuid.UUID) -> None:
        """Hard-delete a session row (CASCADE removes its chats too).

        The user_id check ensures a user can only delete their own sessions.
        """
        conn = self._connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    DELETE FROM memory.sessions
                    WHERE session_id = %s AND user_id = %s;
                    """,
                    (str(session_id), str(user_id)),
                )
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"DB error deleting session: {e}")
                raise MemoryRepositoryError(f"Database error: {e}")
            finally:
                cur.close()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Chat messages
    # ------------------------------------------------------------------

    def add_message(self, session_id: uuid.UUID, sender: str, message: str) -> ChatRecord:
        """Persist a chat message."""
        conn = self._connect()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            try:
                cur.execute(
                    """
                    INSERT INTO memory.chats (session_id, sender, message)
                    VALUES (%s, %s, %s)
                    RETURNING chat_id, session_id, sender, message, created_at;
                    """,
                    (str(session_id), sender, message),
                )
                row = cur.fetchone()
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"DB error adding message: {e}")
                raise MemoryRepositoryError(f"Database error: {e}")
            finally:
                cur.close()
        finally:
            conn.close()

        return ChatRecord(
            chat_id=row["chat_id"],
            session_id=row["session_id"],
            sender=row["sender"],
            message=row["message"],
            created_at=row["created_at"],
        )

    def get_conversation_history(
        self, session_id: uuid.UUID, limit: Optional[int] = None
    ) -> List[ChatRecord]:
        """Fetch chat history for a session in chronological order.

        Args:
            session_id: UUID of the session.
            limit: If set, return only the last N messages.

        Returns:
            List of ChatRecord objects ordered oldest-first.
        """
        conn = self._connect()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            try:
                if limit:
                    cur.execute(
                        """
                        SELECT chat_id, session_id, sender, message, created_at
                        FROM (
                            SELECT * FROM memory.chats
                            WHERE session_id = %s
                            ORDER BY created_at DESC
                            LIMIT %s
                        ) sub
                        ORDER BY created_at ASC;
                        """,
                        (str(session_id), limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT chat_id, session_id, sender, message, created_at
                        FROM memory.chats
                        WHERE session_id = %s
                        ORDER BY created_at ASC;
                        """,
                        (str(session_id),),
                    )
                rows = cur.fetchall()
            except Exception as e:
                logger.error(f"DB error fetching conversation history: {e}")
                raise MemoryRepositoryError(f"Database error: {e}")
            finally:
                cur.close()
        finally:
            conn.close()

        return [
            ChatRecord(
                chat_id=row["chat_id"],
                session_id=row["session_id"],
                sender=row["sender"],
                message=row["message"],
                created_at=row["created_at"],
            )
            for row in rows
        ]