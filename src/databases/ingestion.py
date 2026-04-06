"""PostgreSQL vector ingestion repository.

Concrete implementation of BaseIngestionRepository using asyncpg.
Writes parent chunks to poc2prod.parenthierarchy and child chunks
(with their embeddings) to poc2prod.ingestions.

Usage:
    from src.databases import PgVectorIngestionRepository
    from src.core.config import ConfigManager

    cfg = ConfigManager()
    repo = PgVectorIngestionRepository(cfg.db_config)

    parent_uuids, child_uuids = await repo.ingest_documents(
        parent_docs, child_docs, embeddings,
        user_id=user_id, session_id=session_id,
        filename="paper.pdf", file_description="Research paper", file_type="pdf",
    )
"""

import json
import uuid
from typing import Optional

import asyncpg
from langchain_core.documents import Document

from ..core.exceptions import ResearchPaperChatException
from ..core.logging import LoggingManager
from ..core.models import DBConfig
from .base import BaseIngestionRepository

logger = LoggingManager.get_logger(__name__)

_STRIP_META_KEYS = {"parent_id", "parent_text"}


class IngestionRepositoryError(ResearchPaperChatException):
    """Raised when an ingestion repository operation fails."""
    pass


class PgVectorIngestionRepository(BaseIngestionRepository):
    """Persists hierarchical chunks and embeddings to PostgreSQL (poc2prod schema).

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

    async def ingest_documents(
        self,
        parent_docs: list[Document],
        child_docs: list[Document],
        embeddings: list[list[float]],
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        filename: str,
        file_description: str,
        file_type: str,
        content_type: str = "text",
    ) -> tuple[list[str], list[str]]:
        """Insert parent chunks then child chunks in a single connection.

        Parent chunks are inserted first to obtain their DB UUIDs, which are
        then used to populate the parent_id FK on each child row.

        Metadata keys 'parent_id' and 'parent_text' (set by HierarchicalChunker)
        are stripped before storing child metadata — they are represented by the
        FK column and the parenthierarchy table respectively.
        """
        if len(embeddings) != len(child_docs):
            raise ValueError(
                f"embeddings length ({len(embeddings)}) must match "
                f"child_docs length ({len(child_docs)})"
            )

        conn = await self._connect()
        try:
            # ----------------------------------------------------------------
            # 1. Insert parent chunks → collect int-index → DB UUID mapping
            # ----------------------------------------------------------------
            parent_uuids: list[str] = []
            for doc in parent_docs:
                parent_meta = {k: v for k, v in doc.metadata.items()}
                row = await conn.fetchrow(
                    """
                    INSERT INTO poc2prod.parenthierarchy
                        (parent_chunk_content, filename, metadata, content_type)
                    VALUES ($1, $2, $3::jsonb, $4)
                    RETURNING id;
                    """,
                    doc.page_content,
                    filename,
                    json.dumps(parent_meta),
                    content_type,
                )
                parent_uuids.append(str(row["id"]))

            parent_index_to_uuid: dict[int, str] = {
                i: uid for i, uid in enumerate(parent_uuids)
            }

            # ----------------------------------------------------------------
            # 2. Insert child chunks with embeddings
            # ----------------------------------------------------------------
            child_uuids: list[str] = []
            for doc, vector in zip(child_docs, embeddings):
                meta = doc.metadata
                parent_int_id: Optional[int] = meta.get("parent_id")
                parent_uuid = (
                    parent_index_to_uuid.get(parent_int_id)
                    if parent_int_id is not None
                    else None
                )

                stored_meta = {k: v for k, v in meta.items() if k not in _STRIP_META_KEYS}

                row = await conn.fetchrow(
                    """
                    INSERT INTO poc2prod.ingestions
                        (parent_id, user_id, session_id, filename, file_description,
                         type, chunk_content, embeddings, metadata, content_type)
                    VALUES
                        ($1, $2, $3, $4, $5, $6, $7, $8::vector, $9::jsonb, $10)
                    RETURNING id;
                    """,
                    parent_uuid,
                    str(user_id),
                    str(session_id),
                    filename,
                    file_description,
                    file_type,
                    doc.page_content,
                    self._vec_str(vector),
                    json.dumps(stored_meta),
                    content_type,
                )
                child_uuids.append(str(row["id"]))

        except IngestionRepositoryError:
            raise
        except Exception as e:
            logger.error(f"Ingestion failed for '{filename}': {e}")
            raise IngestionRepositoryError(f"Ingestion failed: {e}") from e
        finally:
            await conn.close()

        logger.info(
            f"Ingested {len(parent_uuids)} parents, {len(child_uuids)} children "
            f"for '{filename}' (user={user_id}, session={session_id})"
        )
        return parent_uuids, child_uuids
