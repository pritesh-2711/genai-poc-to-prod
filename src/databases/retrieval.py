"""PostgreSQL vector retrieval repository.

Concrete implementation of BaseRetrievalRepository using asyncpg.
Performs cosine similarity search over poc2prod.ingestions and fetches
full parent context from poc2prod.parenthierarchy.

Usage:
    from src.databases import PgVectorRetrievalRepository
    from src.core.config import ConfigManager

    cfg = ConfigManager()
    repo = PgVectorRetrievalRepository(cfg.db_config)

    results = await repo.search(query_embedding, top_k=5)
    parent_ids = [r["parent_id"] for r in results if r["parent_id"]]
    contexts  = await repo.fetch_parent_contexts(parent_ids)
"""

import json
import uuid
from typing import Optional

import asyncpg

from ..core.exceptions import ResearchPaperChatException
from ..core.logging import LoggingManager
from ..core.models import DBConfig
from .base import BaseRetrievalRepository

logger = LoggingManager.get_logger(__name__)


class RetrievalRepositoryError(ResearchPaperChatException):
    """Raised when a retrieval repository operation fails."""
    pass


class PgVectorRetrievalRepository(BaseRetrievalRepository):
    """Queries the vector store for semantically similar chunks.

    All public methods are coroutines — use with await inside an async context.
    Each method opens and closes its own connection via try/finally to prevent
    connection leaks on exceptions.
    """

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
        """Serialize a float vector to the pgvector literal format: '[v1,v2,...]'"""
        return "[" + ",".join(str(v) for v in vector) + "]"

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        session_id: Optional[uuid.UUID] = None,
    ) -> list[dict]:
        """Cosine similarity search over ingested child chunks.

        Uses the pgvector <=> operator (cosine distance). Similarity is
        returned as (1 - distance) so higher values mean more similar.

        Args:
            query_embedding: Float vector for the search query.
            top_k:           Number of nearest neighbours to return.
            session_id:      If given, restrict search to this session's chunks.

        Returns:
            List of result dicts ordered by similarity descending:
                child_id       (str)
                parent_id      (str | None)
                chunk_content  (str)
                filename       (str)
                metadata       (dict)
                similarity     (float)
        """
        vec_str = self._vec_str(query_embedding)

        if session_id is not None:
            sql = """
                SELECT
                    id::text                                AS child_id,
                    parent_id::text                        AS parent_id,
                    chunk_content,
                    filename,
                    metadata,
                    1 - (embeddings <=> $1::vector)        AS similarity
                FROM poc2prod.ingestions
                WHERE session_id = $2
                ORDER BY embeddings <=> $1::vector
                LIMIT $3;
            """
            params = (vec_str, str(session_id), top_k)
        else:
            sql = """
                SELECT
                    id::text                                AS child_id,
                    parent_id::text                        AS parent_id,
                    chunk_content,
                    filename,
                    metadata,
                    1 - (embeddings <=> $1::vector)        AS similarity
                FROM poc2prod.ingestions
                ORDER BY embeddings <=> $1::vector
                LIMIT $2;
            """
            params = (vec_str, top_k)

        conn = await self._connect()
        try:
            rows = await conn.fetch(sql, *params)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise RetrievalRepositoryError(f"Vector search failed: {e}") from e
        finally:
            await conn.close()

        results = []
        for row in rows:
            meta = row["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            results.append({
                "child_id": row["child_id"],
                "parent_id": row["parent_id"],
                "chunk_content": row["chunk_content"],
                "filename": row["filename"],
                "metadata": meta,
                "similarity": float(row["similarity"]),
            })

        logger.debug(f"search() returned {len(results)} results (top_k={top_k})")
        return results

    async def fetch_parent_contexts(
        self,
        parent_ids: list[str],
    ) -> list[dict]:
        """Fetch full parent chunk content by UUID list.

        Args:
            parent_ids: List of parenthierarchy UUIDs (as strings).

        Returns:
            List of dicts (order matches DB row order, not input order):
                id                   (str)
                parent_chunk_content (str)
                filename             (str)
                metadata             (dict)
        """
        if not parent_ids:
            return []

        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT
                    id::text                AS id,
                    parent_chunk_content,
                    filename,
                    metadata
                FROM poc2prod.parenthierarchy
                WHERE id = ANY($1::uuid[]);
                """,
                parent_ids,
            )
        except Exception as e:
            logger.error(f"fetch_parent_contexts failed: {e}")
            raise RetrievalRepositoryError(f"fetch_parent_contexts failed: {e}") from e
        finally:
            await conn.close()

        results = []
        for row in rows:
            meta = row["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            results.append({
                "id": row["id"],
                "parent_chunk_content": row["parent_chunk_content"],
                "filename": row["filename"],
                "metadata": meta,
            })

        return results

    async def fetch_colocated_chunks(
        self,
        session_id: uuid.UUID,
        pages: list[int],
        filenames: list[str],
        content_types: tuple[str, ...] = ("table", "image"),
    ) -> list[dict]:
        """Fetch table/image chunks that are co-located (same page + file) with
        the parent contexts already retrieved for a query.

        Called after fetch_parent_contexts so that any tables or figures on the
        same pages as the matched text passages are surfaced automatically,
        regardless of their own vector similarity score.

        Args:
            session_id:    Restrict to this session's ingested chunks.
            pages:         Page numbers covered by the fetched parent contexts.
            filenames:     Filenames of those parent contexts.
            content_types: Which non-text content types to include.

        Returns:
            List of dicts with keys: chunk_content, filename, metadata,
            content_type.
        """
        if not pages or not filenames:
            return []

        page_strs = [str(p) for p in pages]

        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT
                    chunk_content,
                    filename,
                    metadata,
                    content_type
                FROM poc2prod.ingestions
                WHERE session_id = $1
                  AND content_type = ANY($2::text[])
                  AND filename = ANY($3::text[])
                  AND metadata->>'page' = ANY($4::text[])
                ORDER BY (metadata->>'page')::int, content_type;
                """,
                str(session_id),
                list(content_types),
                filenames,
                page_strs,
            )
        except Exception as e:
            logger.warning(f"fetch_colocated_chunks failed (non-fatal): {e}")
            return []
        finally:
            await conn.close()

        results = []
        for row in rows:
            meta = row["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            results.append({
                "chunk_content": row["chunk_content"],
                "filename": row["filename"],
                "metadata": meta,
                "content_type": row["content_type"],
            })

        logger.debug(
            f"fetch_colocated_chunks() returned {len(results)} table/image chunks "
            f"for pages {pages}"
        )
        return results
