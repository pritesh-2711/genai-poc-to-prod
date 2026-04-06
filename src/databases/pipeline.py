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
from ..extraction.table import TableExtractor
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
    table_parent_count: int = 0
    table_child_count: int = 0
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
        # Stage 1 — Extract (layout → text + tables)
        # ------------------------------------------------------------------
        try:
            context = ExtractionContext(file_path=str(file_path))
            LayoutExtractor().extract(context)
            text_records = TextExtractor().extract(context)
            try:
                table_records = TableExtractor().extract(context)
            except Exception as te:
                logger.warning(f"Table extraction failed for '{filename}' (skipping tables): {te}")
                table_records = []
        except Exception as e:
            raise IngestionPipelineError(
                f"Extraction failed for '{filename}': {e}"
            ) from e

        # ------------------------------------------------------------------
        # Stage 2a — Chunk text records
        #
        # HierarchicalChunker assigns child parent_id as a LOCAL index
        # (0-based) within each chunk_with_parents() call.  ingest_documents()
        # builds a GLOBAL index → UUID map, so we must offset each call's
        # local indices by the count of parents already accumulated.
        # ------------------------------------------------------------------
        text_parent_docs = []
        text_child_docs = []
        text_parent_offset = 0

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
            # Offset local parent_ids to global indices
            for child in children:
                child.metadata["parent_id"] = (
                    child.metadata["parent_id"] + text_parent_offset
                )
            text_parent_docs.extend(parents)
            text_child_docs.extend(children)
            text_parent_offset += len(parents)

        # ------------------------------------------------------------------
        # Stage 2b — Chunk table records (markdown representation)
        # ------------------------------------------------------------------
        table_parent_docs = []
        table_child_docs = []
        table_parent_offset = 0

        for record in table_records:
            markdown = (record.content or {}).get("markdown", "")
            if not markdown or not markdown.strip():
                continue

            table_text = f"Table (page {record.page}):\n{markdown}"
            meta = {"page": record.page, "type": "table"}
            if record.bbox:
                meta["bbox"] = record.bbox

            parents, children = self._chunker.chunk_with_parents(
                table_text, metadata=meta
            )
            for child in children:
                child.metadata["parent_id"] = (
                    child.metadata["parent_id"] + table_parent_offset
                )
            table_parent_docs.extend(parents)
            table_child_docs.extend(children)
            table_parent_offset += len(parents)

        if not text_child_docs and not table_child_docs:
            logger.warning(f"No extractable content in '{filename}' — skipping ingestion")
            return IngestionResult(filename=filename, parent_count=0, child_count=0)

        # ------------------------------------------------------------------
        # Stage 3 — Embed all child chunks
        # ------------------------------------------------------------------
        try:
            all_texts = (
                [doc.page_content for doc in text_child_docs]
                + [doc.page_content for doc in table_child_docs]
            )
            all_embeddings = self._embedder.embed(all_texts)
        except Exception as e:
            raise IngestionPipelineError(
                f"Embedding failed for '{filename}': {e}"
            ) from e

        text_embeddings = all_embeddings[: len(text_child_docs)]
        table_embeddings = all_embeddings[len(text_child_docs):]

        # ------------------------------------------------------------------
        # Stage 4 — Persist text chunks
        # ------------------------------------------------------------------
        text_parent_uuids: list[str] = []
        text_child_uuids: list[str] = []

        if text_child_docs:
            text_parent_uuids, text_child_uuids = await self._repo.ingest_documents(
                parent_docs=text_parent_docs,
                child_docs=text_child_docs,
                embeddings=text_embeddings,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
                file_description=file_description,
                file_type=file_type,
                content_type="text",
            )

        # ------------------------------------------------------------------
        # Stage 4b — Persist table chunks
        # ------------------------------------------------------------------
        table_parent_uuids: list[str] = []
        table_child_uuids: list[str] = []

        if table_child_docs:
            table_parent_uuids, table_child_uuids = await self._repo.ingest_documents(
                parent_docs=table_parent_docs,
                child_docs=table_child_docs,
                embeddings=table_embeddings,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
                file_description=file_description,
                file_type=file_type,
                content_type="table",
            )

        total_parents = len(text_parent_uuids) + len(table_parent_uuids)
        total_children = len(text_child_uuids) + len(table_child_uuids)

        logger.info(
            f"Pipeline complete for '{filename}': "
            f"{len(text_parent_uuids)} text parents, {len(text_child_uuids)} text children; "
            f"{len(table_parent_uuids)} table parents, {len(table_child_uuids)} table children "
            f"(user={user_id}, session={session_id})"
        )
        return IngestionResult(
            filename=filename,
            parent_count=total_parents,
            child_count=total_children,
            table_parent_count=len(table_parent_uuids),
            table_child_count=len(table_child_uuids),
            parent_uuids=text_parent_uuids + table_parent_uuids,
            child_uuids=text_child_uuids + table_child_uuids,
        )
