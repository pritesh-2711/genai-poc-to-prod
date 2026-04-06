"""Document ingestion pipeline.

Orchestrates the full upload-to-vector-store flow for a single file:
    1. Extract text records via LayoutExtractor → TextExtractor
    2. Hierarchically chunk text records (parent + child split)
    3. Embed child chunks with the configured embedder
    4. INSERT parents into parenthierarchy, children into ingestions

The pipeline is intentionally stateless — create one per request or share
the same instance (the embedder holds the only stateful resource).

Usage:
    pipeline = IngestionPipeline(db_config=cfg.db_config, embedder=embedder)
    result = await pipeline.run(
        file_path=saved_path,
        user_id=user_id,
        session_id=session_id,
        file_description="Q3 earnings report",
        file_type="pdf",
    )
    print(result.parent_count, result.child_count)
"""

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..chunker import HierarchicalChunker
from ..core.exceptions import ResearchPaperChatException
from ..core.logging import LoggingManager
from ..core.models import DBConfig
from ..embedding.base import BaseEmbedder
from ..extraction.base import ExtractionContext
from ..extraction.layout import LayoutExtractor
from ..extraction.text import TextExtractor
from .ingestion import PgVectorIngestionRepository

logger = LoggingManager.get_logger(__name__)

_TEXT_TYPES = ("text", "latex")

_CONTENT_TYPE_TO_FILE_TYPE = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "doc",
}


@dataclass
class IngestionResult:
    """Summary of a completed ingestion run."""

    filename: str
    parent_count: int
    child_count: int
    parent_uuids: list[str] = field(default_factory=list)
    child_uuids: list[str] = field(default_factory=list)


class IngestionPipelineError(ResearchPaperChatException):
    """Raised when any stage of the ingestion pipeline fails."""
    pass


class IngestionPipeline:
    """Runs extract → chunk → embed → ingest for a single uploaded file.

    Args:
        db_config: PostgreSQL connection config.
        embedder:  Any BaseEmbedder (Local, Ollama, OpenAI).
        chunker:   HierarchicalChunker instance; a default one is created if
                   not provided.
    """

    def __init__(
        self,
        db_config: DBConfig,
        embedder: BaseEmbedder,
        chunker: HierarchicalChunker | None = None,
    ) -> None:
        self._db_config = db_config
        self._embedder = embedder
        self._chunker = chunker or HierarchicalChunker()
        self._repo = PgVectorIngestionRepository(db_config)

    async def run(
        self,
        file_path: Path | str,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        file_description: str = "",
        file_type: str = "pdf",
    ) -> IngestionResult:
        """Run the full pipeline for a single file.

        Args:
            file_path:        Absolute path to the saved file.
            user_id:          Owning user's UUID.
            session_id:       Owning session's UUID.
            file_description: Human-readable description (stored in ingestions).
            file_type:        "pdf" or "doc" — stored in ingestions.type column.

        Returns:
            IngestionResult with counts and DB-assigned UUIDs.

        Raises:
            IngestionPipelineError: If extraction, embedding, or DB insertion fails.
        """
        file_path = Path(file_path)
        filename = file_path.name

        # ------------------------------------------------------------------
        # Stage 1 — Extract
        # ------------------------------------------------------------------
        try:
            context = ExtractionContext(file_path=str(file_path))
            LayoutExtractor().extract(context)
            text_records = TextExtractor().extract(context)
        except Exception as e:
            raise IngestionPipelineError(
                f"Extraction failed for '{filename}': {e}"
            ) from e

        # ------------------------------------------------------------------
        # Stage 2 — Hierarchical chunking
        # ------------------------------------------------------------------
        all_parent_docs = []
        all_child_docs = []

        for record in text_records:
            if record.record_type not in _TEXT_TYPES:
                continue
            if not isinstance(record.content, str) or not record.content.strip():
                continue

            meta: dict = {"page": record.page, "type": record.record_type}
            if record.bbox:
                meta["bbox"] = record.bbox

            parents, children = self._chunker.chunk_with_parents(
                record.content, metadata=meta
            )
            all_parent_docs.extend(parents)
            all_child_docs.extend(children)

        if not all_child_docs:
            logger.warning(f"No extractable text in '{filename}' — skipping ingestion")
            return IngestionResult(filename=filename, parent_count=0, child_count=0)

        # ------------------------------------------------------------------
        # Stage 3 — Embed child chunks
        # ------------------------------------------------------------------
        try:
            child_texts = [doc.page_content for doc in all_child_docs]
            embeddings = self._embedder.embed(child_texts)
        except Exception as e:
            raise IngestionPipelineError(
                f"Embedding failed for '{filename}': {e}"
            ) from e

        # ------------------------------------------------------------------
        # Stage 4 — Persist to PostgreSQL
        # ------------------------------------------------------------------
        parent_uuids, child_uuids = await self._repo.ingest_documents(
            parent_docs=all_parent_docs,
            child_docs=all_child_docs,
            embeddings=embeddings,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
            file_description=file_description,
            file_type=file_type,
        )

        logger.info(
            f"Pipeline complete for '{filename}': "
            f"{len(parent_uuids)} parents, {len(child_uuids)} children "
            f"(user={user_id}, session={session_id})"
        )
        return IngestionResult(
            filename=filename,
            parent_count=len(parent_uuids),
            child_count=len(child_uuids),
            parent_uuids=parent_uuids,
            child_uuids=child_uuids,
        )
