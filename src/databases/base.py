"""Base abstractions for vector store repositories.

Two ABCs define the ingestion and retrieval interfaces independently:

    BaseIngestionRepository   — insert parent/child chunks + embeddings
    BaseRetrievalRepository   — cosine search + parent context fetch

This mirrors the embedding and chunker module pattern: every concrete
implementation receives a DBConfig in __init__ and is swappable without
changing any consumer code.
"""

import uuid
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.documents import Document

from ..core.models import DBConfig


class BaseIngestionRepository(ABC):
    """Abstract interface for persisting hierarchical chunks to a vector store."""

    def __init__(self, db_config: DBConfig) -> None:
        self.db_config = db_config

    @abstractmethod
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
    ) -> tuple[list[str], list[str]]:
        """Persist parent chunks and embedded child chunks.

        Expects output from HierarchicalChunker.chunk_with_parents():
          - parent_docs : large context chunks stored in parenthierarchy table
          - child_docs  : small indexed chunks stored in ingestions table;
                          each doc's metadata must contain parent_id (int index)
          - embeddings  : one float vector per child_doc, same order

        Args:
            parent_docs:      Parent Document list from chunk_with_parents().
            child_docs:       Child Document list from chunk_with_parents().
            embeddings:       Float vectors for child chunks (len == len(child_docs)).
            user_id:          UUID of the owning user.
            session_id:       UUID of the owning session.
            filename:         Original file name (e.g. "paper.pdf").
            file_description: Human-readable description of the file.
            file_type:        "pdf" or "doc".

        Returns:
            (parent_uuids, child_uuids) — DB-assigned UUIDs in insertion order.

        Raises:
            ValueError: If embeddings length does not match child_docs length.
            IngestionRepositoryError: On any database error.
        """


class BaseRetrievalRepository(ABC):
    """Abstract interface for querying the vector store."""

    def __init__(self, db_config: DBConfig) -> None:
        self.db_config = db_config

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        session_id: Optional[uuid.UUID] = None,
    ) -> list[dict]:
        """Cosine similarity search over ingested child chunks.

        Args:
            query_embedding: Float vector for the search query.
            top_k:           Number of results to return.
            session_id:      If provided, restrict search to this session.

        Returns:
            List of dicts, each containing:
                child_id       (str)   UUID of the matched child chunk
                parent_id      (str)   UUID of its parent chunk (may be None)
                chunk_content  (str)   Child chunk text
                filename       (str)
                metadata       (dict)
                similarity     (float) 1 - cosine_distance, in [0, 1]

        Raises:
            RetrievalRepositoryError: On any database error.
        """

    @abstractmethod
    async def fetch_parent_contexts(
        self,
        parent_ids: list[str],
    ) -> list[dict]:
        """Fetch full parent chunk content by UUID list.

        Args:
            parent_ids: List of parenthierarchy UUIDs (as strings).

        Returns:
            List of dicts, each containing:
                id              (str)  UUID
                parent_chunk_content (str)
                filename        (str)
                metadata        (dict)

        Raises:
            RetrievalRepositoryError: On any database error.
        """
