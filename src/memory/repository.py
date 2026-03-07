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

    Each method opens and closes its own connection. This keeps things simple
    and avoids connection-lifetime issues in an async Chainlit context.
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
    # Authentication
    # ------------------------------------------------------------------

    def authenticate_user(self, email: str, password: str) -> UserRecord:
        """Verify credentials, update last_login_at, and return the user record.

        Args:
            email: User email address.
            password: Plain-text password supplied by the user.

        Returns:
            UserRecord for the authenticated user.

        Raises:
            AuthenticationError: If credentials are invalid.
            MemoryRepositoryError: On database error.
        """
        try:
            conn = self._connect()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT user_id, name, email, password, created_at FROM memory.users WHERE email = %s;",
                (email,),
            )
            row = cur.fetchone()
        except Exception as e:
            logger.error(f"DB error during authentication: {e}")
            raise MemoryRepositoryError(f"Database error: {e}")

        if row is None:
            cur.close()
            conn.close()
            raise AuthenticationError("Invalid email or password.")

        if not bcrypt.checkpw(password.encode("utf-8"), row["password"].encode("utf-8")):
            cur.close()
            conn.close()
            raise AuthenticationError("Invalid email or password.")

        # Update last_login_at on successful login
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
        try:
            conn = self._connect()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """
                SELECT session_id, user_id, session_name, is_active, created_at
                FROM memory.sessions
                WHERE user_id = %s
                ORDER BY created_at DESC;
                """,
                (str(user_id),),
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"DB error fetching sessions: {e}")
            raise MemoryRepositoryError(f"Database error: {e}")

        return [
            SessionRecord(
                session_id=row["session_id"],
                user_id=row["user_id"],
                session_name=row["session_name"],
                is_active=row["is_active"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def create_session(self, user_id: uuid.UUID, session_name: str) -> SessionRecord:
        """Create a new session for a user."""
        try:
            conn = self._connect()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """
                INSERT INTO memory.sessions (user_id, session_name)
                VALUES (%s, %s)
                RETURNING session_id, user_id, session_name, is_active, created_at;
                """,
                (str(user_id), session_name),
            )
            row = cur.fetchone()
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"DB error creating session: {e}")
            raise MemoryRepositoryError(f"Database error: {e}")

        logger.info(f"Session created: {row['session_id']} for user {user_id}")
        return SessionRecord(
            session_id=row["session_id"],
            user_id=row["user_id"],
            session_name=row["session_name"],
            is_active=row["is_active"],
            created_at=row["created_at"],
        )

    def terminate_session(self, session_id: uuid.UUID) -> None:
        """Mark a session as inactive and stamp terminated_at."""
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE memory.sessions
                SET is_active = FALSE, terminated_at = CURRENT_TIMESTAMP
                WHERE session_id = %s;
                """,
                (str(session_id),),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"DB error terminating session: {e}")
            raise MemoryRepositoryError(f"Database error: {e}")

    # ------------------------------------------------------------------
    # Chat messages
    # ------------------------------------------------------------------

    def add_message(self, session_id: uuid.UUID, sender: str, message: str) -> ChatRecord:
        """Persist a chat message."""
        try:
            conn = self._connect()
            cur = conn.cursor(cursor_factory=RealDictCursor)
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
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"DB error adding message: {e}")
            raise MemoryRepositoryError(f"Database error: {e}")

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
        try:
            conn = self._connect()
            cur = conn.cursor(cursor_factory=RealDictCursor)

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
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"DB error fetching conversation history: {e}")
            raise MemoryRepositoryError(f"Database error: {e}")

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