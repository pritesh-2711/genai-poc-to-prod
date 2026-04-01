"""Hierarchical chunking — parent/child split.

Creates two levels of chunks from each text:

    Parent chunks  (large, 2000 chars)  — stored in docstore / returned to LLM
    Child chunks   (small,  400 chars)  — indexed in vector store for retrieval

Each child Document carries parent_id and parent_text in its metadata so the
retrieval layer can look up the full parent context when a child is matched.

Usage:
    from src.chunker.hierarchical import HierarchicalChunker

    chunker = HierarchicalChunker()
    child_docs = chunker.chunk("long document text...")

    # Each child doc metadata:
    #   parent_id   → int index of the parent chunk
    #   parent_text → full parent chunk content
    #   source, page, type → forwarded from caller

Pipeline usage:
    child_docs = chunker.chunk_records(text_records, source=pdf_path)
"""

from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunker


class HierarchicalChunker(BaseChunker):
    """Splits text into parent chunks then further into child chunks.

    Indexes child chunks in the vector store (dense retrieval on small,
    precise windows) but returns the full parent chunk to the LLM for
    richer context.

    Args:
        parent_chunk_size:    Target size of parent chunks in characters.
        parent_chunk_overlap: Overlap between consecutive parent chunks.
        child_chunk_size:     Target size of child chunks in characters.
        child_chunk_overlap:  Overlap between consecutive child chunks.
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50,
    ) -> None:
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
        )

    def chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[Document]:
        """Split text hierarchically and return child Documents.

        Args:
            text:     Text to chunk.
            metadata: Base metadata merged into every child Document.

        Returns:
            List of child Document objects. Each carries:
                parent_id   — index of the parent chunk this child belongs to
                parent_text — full content of the parent chunk
                + any keys passed via metadata
        """
        base_meta = metadata or {}
        parent_chunks = self._parent_splitter.create_documents([text])

        child_docs: list[Document] = []
        for parent_id, parent in enumerate(parent_chunks):
            children = self._child_splitter.create_documents(
                [parent.page_content],
                metadatas=[{
                    **base_meta,
                    "parent_id": parent_id,
                    "parent_text": parent.page_content,
                }],
            )
            child_docs.extend(children)

        return child_docs

    def chunk_with_parents(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> tuple[list[Document], list[Document]]:
        """Return both parent and child Documents.

        Useful when you want to store parents in a docstore explicitly
        alongside indexing children in the vector store.

        Returns:
            (parent_docs, child_docs)
        """
        base_meta = metadata or {}
        parent_chunks = self._parent_splitter.create_documents(
            [text],
            metadatas=[{**base_meta}],
        )

        child_docs: list[Document] = []
        for parent_id, parent in enumerate(parent_chunks):
            children = self._child_splitter.create_documents(
                [parent.page_content],
                metadatas=[{
                    **base_meta,
                    "parent_id": parent_id,
                    "parent_text": parent.page_content,
                }],
            )
            child_docs.extend(children)

        return parent_chunks, child_docs
