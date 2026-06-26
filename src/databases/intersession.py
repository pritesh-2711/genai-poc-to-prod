"""Async repository for intersession memory and RLHF chunk-score recomputation."""

import logging
import uuid
from typing import Optional

import asyncpg

from ..core.models import DBConfig

logger = logging.getLogger(__name__)


class IntersessionRepository:
    """Manages session summaries (intersession memory) and chunk quality scores.

    Uses asyncpg for all operations — call from async context only.
    Each public method opens and closes its own connection to prevent leaks.
    """

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

    @staticmethod
    def _vec_str(vector: list[float]) -> str:
        return "[" + ",".join(str(v) for v in vector) + "]"

    # ── Session summaries ─────────────────────────────────────────────────────

    async def upsert_session_summary(
        self,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        summary_text: str,
        embedding: list[float],
        token_count: int,
    ) -> None:
        """Insert or refresh the summary row for a session."""
        vec_str = self._vec_str(embedding)
        conn = await self._connect()
        try:
            await conn.execute(
                """
                INSERT INTO poc2prod.session_summaries
                    (user_id, session_id, summary_text, summary_embedding, token_count)
                VALUES ($1, $2, $3, $4::vector, $5)
                ON CONFLICT (session_id) DO UPDATE SET
                    summary_text      = EXCLUDED.summary_text,
                    summary_embedding = EXCLUDED.summary_embedding,
                    token_count       = EXCLUDED.token_count,
                    updated_at        = NOW();
                """,
                str(user_id), str(session_id), summary_text, vec_str, token_count,
            )
        finally:
            await conn.close()

    async def get_relevant_summaries(
        self,
        user_id: uuid.UUID,
        query_embedding: list[float],
        exclude_session_id: uuid.UUID,
        top_k: int = 5,
    ) -> list[dict]:
        """Return up to top_k previous-session summaries for a user, ranked by
        cosine similarity to query_embedding, excluding the current session."""
        vec_str = self._vec_str(query_embedding)
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT
                    session_id::text                              AS session_id,
                    summary_text,
                    token_count,
                    1 - (summary_embedding <=> $1::vector)        AS similarity
                FROM poc2prod.session_summaries
                WHERE user_id = $2
                  AND session_id != $3
                  AND summary_embedding IS NOT NULL
                ORDER BY summary_embedding <=> $1::vector
                LIMIT $4;
                """,
                vec_str, str(user_id), str(exclude_session_id), top_k,
            )
        finally:
            await conn.close()

        return [
            {
                "session_id": row["session_id"],
                "summary_text": row["summary_text"],
                "token_count": int(row["token_count"]),
                "similarity": float(row["similarity"]),
            }
            for row in rows
        ]

    async def get_sessions_for_summary(self) -> list[dict]:
        """Return (session_id, user_id) for every session that has at least one chat."""
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT DISTINCT
                    s.session_id::text  AS session_id,
                    s.user_id::text     AS user_id
                FROM poc2prod.sessions s
                JOIN poc2prod.chats c ON c.session_id = s.session_id
                ORDER BY s.session_id;
                """
            )
        finally:
            await conn.close()

        return [{"session_id": row["session_id"], "user_id": row["user_id"]} for row in rows]

    async def get_session_chat_history_text(self, session_id: uuid.UUID) -> str:
        """Fetch all chats for a session and return them as a single dialogue string."""
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT sender, message
                FROM poc2prod.chats
                WHERE session_id = $1
                ORDER BY created_at ASC;
                """,
                str(session_id),
            )
        finally:
            await conn.close()

        lines = [
            f"{'User' if r['sender'] == 'user' else 'Assistant'}: {r['message']}"
            for r in rows
        ]
        return "\n".join(lines)

    # ── Chunk quality scores ──────────────────────────────────────────────────

    async def recompute_chunk_scores(self) -> int:
        """Recompute quality scores for all chunks with at least one feedback vote.

        Score = (positive + 1) / (positive + negative + 2)  [Laplace smoothing]
        Returns the number of rows updated.
        """
        conn = await self._connect()
        try:
            result = await conn.execute(
                """
                UPDATE poc2prod.chunk_scores
                SET
                    score      = (positive_count + 1.0) / (positive_count + negative_count + 2.0),
                    updated_at = NOW()
                WHERE positive_count > 0 OR negative_count > 0;
                """
            )
        finally:
            await conn.close()

        # asyncpg returns "UPDATE N" as the status string
        try:
            updated = int(result.split()[-1])
        except (AttributeError, ValueError, IndexError):
            updated = 0

        logger.info(f"[chunk_scoring] recomputed scores for {updated} chunks")
        return updated
