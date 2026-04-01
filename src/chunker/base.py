"""Base abstraction for all chunkers.

Every chunker exposes two methods:

    chunk(text, metadata)       — chunk a single string
    chunk_records(records, ...) — chunk ExtractedRecord objects directly
                                  from the extraction pipeline

This makes chunkers pipeline-aware: they understand ExtractedRecord and
filter to text/latex by default, carrying page + source metadata forward
into every LangChain Document they produce.

Flow:
    loader → extraction (ExtractedRecord list) → chunker (Document list)
"""

from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.documents import Document

from ..extraction.base import ExtractedRecord


class BaseChunker(ABC):
    """Abstract base for all chunkers."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[Document]:
        """Chunk a single text string into LangChain Documents.

        Args:
            text:     The text to chunk.
            metadata: Optional dict merged into every produced Document's
                      metadata.

        Returns:
            List of LangChain Document objects.
        """

    def chunk_records(
        self,
        records: list[ExtractedRecord],
        source: str = "",
        include_types: tuple[str, ...] = ("text", "latex"),
    ) -> list[Document]:
        """Chunk a list of ExtractedRecord objects from the extraction pipeline.

        Only records whose record_type is in include_types and whose content
        is a non-empty string are chunked.  Page number, source path, and
        record type are forwarded as metadata into every produced Document.

        Args:
            records:       Output of TextExtractor.extract() or
                           context.layout.records.
            source:        Source identifier (e.g. file path or filename)
                           to tag in metadata.
            include_types: Record types to include. Defaults to text + latex.

        Returns:
            Flat list of LangChain Document objects across all records.
        """
        docs: list[Document] = []
        for record in records:
            if record.record_type not in include_types:
                continue
            if not isinstance(record.content, str) or not record.content.strip():
                continue

            meta: dict = {
                "source": source,
                "page": record.page,
                "type": record.record_type,
            }
            if record.bbox:
                meta["bbox"] = record.bbox

            docs.extend(self.chunk(record.content, meta))

        return docs
