"""Async repository for admin dashboard queries.

Covers every data surface the admin API needs:
  - User management  (list, approve, reject, pending queue)
  - Conversation oversight  (all sessions + message drill-down)
  - Feedback & RLHF stats  (aggregate counts, chunk score table)
  - Governance flags  (output guardrail results, write + read)
  - Knowledge base  (documents across all users, delete)
  - Overview stats  (summary counters for the dashboard landing card)

Uses asyncpg throughout — call from async context only.
Each public method opens and closes its own connection.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import asyncpg

from ..core.models import DBConfig

logger = logging.getLogger(__name__)


class AdminRepository:
    def __init__(self, db_config: DBConfig) -> None:
        self.db_config = db_config

    async def _connect(self) -> asyncpg.Connection:
        return await asyncpg.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            database=self.db_config.database,
            user=self.db_config.user,
            password=self.db_config.password,
            server_settings={"search_path": "poc2prod,public"},
        )

    # ── Overview ──────────────────────────────────────────────────────────────

    async def get_overview_stats(self) -> dict:
        """Return the four headline counters shown on the Overview card."""
        conn = await self._connect()
        try:
            pending = await conn.fetchval(
                "SELECT COUNT(*) FROM poc2prod.users WHERE status = 'pending';"
            )
            flagged = await conn.fetchval(
                "SELECT COUNT(*) FROM poc2prod.governance_flags WHERE flagged = TRUE;"
            )
            active_users = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT user_id) FROM poc2prod.sessions
                WHERE created_at >= NOW() - INTERVAL '7 days';
                """
            )
        finally:
            await conn.close()

        return {
            "pending_approvals": int(pending or 0),
            "flagged_responses": int(flagged or 0),
            "active_users_7d": int(active_users or 0),
        }

    async def get_recent_activity(self, limit: int = 10) -> list[dict]:
        """Return the most recent admin-relevant events across all tables."""
        conn = await self._connect()
        try:
            signups = await conn.fetch(
                """
                SELECT
                    'signup'      AS event_type,
                    name          AS detail,
                    created_at    AS occurred_at
                FROM poc2prod.users
                ORDER BY created_at DESC
                LIMIT $1;
                """,
                limit,
            )
            flags = await conn.fetch(
                """
                SELECT
                    'flagged'     AS event_type,
                    flag_reason   AS detail,
                    created_at    AS occurred_at
                FROM poc2prod.governance_flags
                WHERE flagged = TRUE
                ORDER BY created_at DESC
                LIMIT $1;
                """,
                limit,
            )
        finally:
            await conn.close()

        events = [
            {"event_type": r["event_type"], "detail": r["detail"], "occurred_at": r["occurred_at"].isoformat()}
            for r in list(signups) + list(flags)
        ]
        events.sort(key=lambda e: e["occurred_at"], reverse=True)
        return events[:limit]

    # ── User management ───────────────────────────────────────────────────────

    async def get_pending_users(self) -> list[dict]:
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT user_id::text, name, email, created_at
                FROM poc2prod.users
                WHERE status = 'pending'
                ORDER BY created_at ASC;
                """
            )
        finally:
            await conn.close()
        return [dict(r) for r in rows]

    async def get_all_users(self, search: Optional[str] = None, limit: int = 50, offset: int = 0) -> list[dict]:
        conn = await self._connect()
        try:
            if search:
                rows = await conn.fetch(
                    """
                    SELECT user_id::text, name, email, status, created_at, last_login_at
                    FROM poc2prod.users
                    WHERE name ILIKE $1 OR email ILIKE $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3;
                    """,
                    f"%{search}%", limit, offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT user_id::text, name, email, status, created_at, last_login_at
                    FROM poc2prod.users
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2;
                    """,
                    limit, offset,
                )
        finally:
            await conn.close()
        return [dict(r) for r in rows]

    async def set_user_status(self, user_id: uuid.UUID, status: str) -> bool:
        """Approve or reject a user.  Returns True if a row was updated."""
        conn = await self._connect()
        try:
            result = await conn.execute(
                "UPDATE poc2prod.users SET status = $1, updated_at = NOW() WHERE user_id = $2;",
                status, str(user_id),
            )
        finally:
            await conn.close()
        return result == "UPDATE 1"

    # ── Conversations ─────────────────────────────────────────────────────────

    async def get_all_sessions(
        self,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """All sessions across all users, with user email and message count."""
        conn = await self._connect()
        try:
            base = """
                SELECT
                    s.session_id::text,
                    u.email              AS user_email,
                    s.session_name,
                    s.is_active,
                    s.created_at,
                    COUNT(c.chat_id)     AS message_count,
                    MAX(c.orchestrator_metadata->>'mode') AS last_mode
                FROM poc2prod.sessions s
                JOIN poc2prod.users u ON u.user_id = s.user_id
                LEFT JOIN poc2prod.chats c ON c.session_id = s.session_id
            """
            if search:
                rows = await conn.fetch(
                    base + """
                    WHERE u.email ILIKE $1 OR s.session_id::text ILIKE $1
                    GROUP BY s.session_id, u.email
                    ORDER BY s.created_at DESC
                    LIMIT $2 OFFSET $3;
                    """,
                    f"%{search}%", limit, offset,
                )
            else:
                rows = await conn.fetch(
                    base + """
                    GROUP BY s.session_id, u.email
                    ORDER BY s.created_at DESC
                    LIMIT $1 OFFSET $2;
                    """,
                    limit, offset,
                )
        finally:
            await conn.close()
        return [dict(r) for r in rows]

    async def get_session_messages(self, session_id: uuid.UUID) -> list[dict]:
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT chat_id::text, sender, message, created_at,
                       orchestrator_metadata
                FROM poc2prod.chats
                WHERE session_id = $1
                ORDER BY created_at ASC;
                """,
                str(session_id),
            )
        finally:
            await conn.close()
        return [
            {
                "chat_id": r["chat_id"],
                "sender": r["sender"],
                "message": r["message"],
                "created_at": r["created_at"].isoformat(),
                "orchestrator_metadata": r["orchestrator_metadata"] or {},
            }
            for r in rows
        ]

    async def get_session_summaries(self, limit: int = 20) -> list[dict]:
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT
                    ss.session_id::text,
                    u.email         AS user_email,
                    ss.summary_text,
                    ss.token_count,
                    ss.updated_at
                FROM poc2prod.session_summaries ss
                JOIN poc2prod.users u ON u.user_id = ss.user_id
                ORDER BY ss.updated_at DESC
                LIMIT $1;
                """,
                limit,
            )
        finally:
            await conn.close()
        return [dict(r) for r in rows]

    # ── Feedback & RLHF ───────────────────────────────────────────────────────

    async def get_feedback_stats(self) -> dict:
        conn = await self._connect()
        try:
            total_7d = await conn.fetchval(
                "SELECT COUNT(*) FROM poc2prod.feedback WHERE created_at >= NOW() - INTERVAL '7 days';"
            )
            positive_7d = await conn.fetchval(
                "SELECT COUNT(*) FROM poc2prod.feedback WHERE rating = 'up' AND created_at >= NOW() - INTERVAL '7 days';"
            )
        finally:
            await conn.close()

        total = int(total_7d or 0)
        positive = int(positive_7d or 0)
        positive_rate = round(positive / total * 100) if total > 0 else 0
        return {"ratings_7d": total, "positive_rate": positive_rate}

    async def get_chunk_scores(self, limit: int = 50) -> list[dict]:
        """Chunk scores with document filename, sorted by score ascending (worst first)."""
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT
                    cs.chunk_id::text,
                    i.filename,
                    cs.positive_count,
                    cs.negative_count,
                    cs.score,
                    cs.updated_at
                FROM poc2prod.chunk_scores cs
                JOIN poc2prod.ingestions i ON i.id = cs.chunk_id
                ORDER BY cs.score ASC
                LIMIT $1;
                """,
                limit,
            )
        finally:
            await conn.close()
        return [dict(r) for r in rows]

    # ── Governance flags ──────────────────────────────────────────────────────

    async def get_governance_flags(self, only_flagged: bool = False, limit: int = 50) -> list[dict]:
        conn = await self._connect()
        try:
            where = "WHERE gf.flagged = TRUE" if only_flagged else ""
            rows = await conn.fetch(
                f"""
                SELECT
                    gf.id::text,
                    gf.chat_id::text,
                    gf.session_id::text,
                    gf.toxicity_score,
                    gf.bias_score,
                    gf.faithfulness_score,
                    gf.flagged,
                    gf.flag_reason,
                    gf.created_at
                FROM poc2prod.governance_flags gf
                {where}
                ORDER BY gf.created_at DESC
                LIMIT $1;
                """,
                limit,
            )
        finally:
            await conn.close()
        return [dict(r) for r in rows]

    async def upsert_governance_flag(
        self,
        chat_id: uuid.UUID,
        session_id: uuid.UUID,
        toxicity_score: float,
        bias_score: float,
        faithfulness_score: Optional[float],
        flagged: bool,
        flag_reason: Optional[str],
    ) -> None:
        conn = await self._connect()
        try:
            await conn.execute(
                """
                INSERT INTO poc2prod.governance_flags
                    (chat_id, session_id, toxicity_score, bias_score,
                     faithfulness_score, flagged, flag_reason)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (chat_id) DO UPDATE SET
                    toxicity_score     = EXCLUDED.toxicity_score,
                    bias_score         = EXCLUDED.bias_score,
                    faithfulness_score = EXCLUDED.faithfulness_score,
                    flagged            = EXCLUDED.flagged,
                    flag_reason        = EXCLUDED.flag_reason,
                    created_at         = NOW();
                """,
                str(chat_id), str(session_id),
                toxicity_score, bias_score, faithfulness_score,
                flagged, flag_reason,
            )
        finally:
            await conn.close()

    async def get_unprocessed_assistant_messages(
        self,
        window_hours: int = 24,
        limit: int = 100,
    ) -> list[dict]:
        """Return recent assistant messages that have no governance_flag yet."""
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT
                    c.chat_id::text,
                    c.session_id::text,
                    c.message,
                    c.orchestrator_metadata,
                    c.created_at
                FROM poc2prod.chats c
                LEFT JOIN poc2prod.governance_flags gf ON gf.chat_id = c.chat_id
                WHERE c.sender = 'assistant'
                  AND gf.chat_id IS NULL
                  AND c.created_at >= NOW() - ($1 || ' hours')::INTERVAL
                ORDER BY c.created_at DESC
                LIMIT $2;
                """,
                str(window_hours), limit,
            )
        finally:
            await conn.close()
        return [
            {
                "chat_id": r["chat_id"],
                "session_id": r["session_id"],
                "message": r["message"],
                "orchestrator_metadata": r["orchestrator_metadata"] or {},
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ]

    async def get_preceding_user_message(self, session_id: uuid.UUID, before_iso: str) -> Optional[str]:
        """Return the most recent user message before a given timestamp in the same session."""
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                """
                SELECT message FROM poc2prod.chats
                WHERE session_id = $1
                  AND sender = 'user'
                  AND created_at < $2::timestamptz
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                str(session_id), before_iso,
            )
        finally:
            await conn.close()
        return row["message"] if row else None

    # ── Knowledge base ────────────────────────────────────────────────────────

    async def get_all_documents(self) -> list[dict]:
        """All ingested documents across all users with chunk counts."""
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT
                    i.filename,
                    COALESCE(MAX(i.file_description), '') AS file_description,
                    COALESCE(MAX(i.type), 'pdf')          AS file_type,
                    COUNT(DISTINCT i.parent_id)           AS parent_chunks,
                    COUNT(i.id)                           AS child_chunks,
                    MIN(i.created_at)                     AS ingested_at
                FROM poc2prod.ingestions i
                GROUP BY i.filename
                ORDER BY MIN(i.created_at) DESC;
                """
            )
        finally:
            await conn.close()
        return [
            {
                "filename": r["filename"],
                "file_description": r["file_description"],
                "file_type": r["file_type"],
                "parent_chunks": int(r["parent_chunks"]),
                "child_chunks": int(r["child_chunks"]),
                "ingested_at": r["ingested_at"].isoformat() if r["ingested_at"] else None,
            }
            for r in rows
        ]

    async def delete_document(self, filename: str) -> int:
        """Delete all ingestion rows (and their parent chunks) for a filename.

        Returns the number of child chunk rows deleted.
        """
        conn = await self._connect()
        try:
            result = await conn.execute(
                "DELETE FROM poc2prod.ingestions WHERE filename = $1;",
                filename,
            )
            deleted_children = int(result.split()[-1])

            await conn.execute(
                "DELETE FROM poc2prod.parenthierarchy WHERE filename = $1;",
                filename,
            )
        finally:
            await conn.close()
        logger.info(f"[admin] deleted {deleted_children} child chunks for filename='{filename}'")
        return deleted_children
