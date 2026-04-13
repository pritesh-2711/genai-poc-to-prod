"""
Document tools that wire into the existing retrieval and extraction infrastructure.

These tools are injected with session context at runtime via a factory function.
See the bottom of this file for build_document_tools(), which the API layer calls
at startup to produce session-scoped tool instances.
"""

import json
from typing import Any
from uuid import UUID

from langchain_core.tools import tool

from ..core.logging import LoggingManager
from ..databases.retrieval import PgVectorRetrievalRepository
from ..embedding.base import BaseEmbedder
from ..memory.repository import MemoryRepository

logger = LoggingManager.get_logger(__name__)

_DEFAULT_TOP_K = 5
_MAX_TOP_K = 10


# ---------------------------------------------------------------------------
# Runtime-injected tool factories
#
# LangChain @tool functions must be plain callables — they cannot receive
# dependencies through FastAPI Depends(). Instead, we use closures: each
# factory captures the session context and returns a @tool-decorated function
# bound to it. The API layer calls build_document_tools() once per request
# and passes the resulting tool list to the agent.
# ---------------------------------------------------------------------------

def build_document_tools(
    embedder: BaseEmbedder,
    retrieval_repo: PgVectorRetrievalRepository,
    memory_repo: MemoryRepository,
    session_id: UUID,
    user_id: UUID,
) -> list[Any]:
    """
    Build the full set of document tools scoped to a specific user session.

    Call this from the chat endpoint after resolving dependencies:

        tools = build_document_tools(
            embedder=embedder,
            retrieval_repo=retrieval_repo,
            memory_repo=memory_repo,
            session_id=session_id,
            user_id=current_user.user_id,
        )

    Args:
        embedder: The active embedding backend (LocalEmbedder, OllamaEmbedder, etc.)
        retrieval_repo: PgVectorRetrievalRepository for the current db config.
        memory_repo: MemoryRepository for conversation and document metadata.
        session_id: UUID of the current chat session.
        user_id: UUID of the authenticated user.

    Returns:
        List of LangChain tool objects ready to be passed to create_agent or bind_tools.
    """

    @tool
    async def search_documents(query: str, top_k: int = _DEFAULT_TOP_K) -> str:
        """Search the documents uploaded to this session using semantic similarity.

        Use this when the user asks something that requires specific content from their
        uploaded research papers — methodology details, experimental results, specific
        claims, equations, table data, or any information that would only appear in
        the documents rather than general knowledge.

        Call this multiple times with different queries if the first search does not
        return sufficient information. Rephrase the query to target different aspects
        of the same question.

        Do NOT use this for general questions about well-known concepts that the LLM
        can answer directly without the documents.

        Args:
            query: A focused search query targeting the specific information needed.
                   Use technical terms from the domain. Avoid vague phrases like
                   "information about the paper" — be specific about what you need.
            top_k: Number of document chunks to retrieve. Default 5, max 10.

        Returns:
            Relevant excerpts from the uploaded documents with source filenames.
        """
        top_k = min(top_k, _MAX_TOP_K)
        logger.info("search_documents: query=%r top_k=%d session=%s", query, top_k, session_id)

        try:
            query_vec = embedder.embed_query(query)
            child_hits = await retrieval_repo.search(
                query_embedding=query_vec,
                top_k=top_k,
                session_id=session_id,
            )
        except Exception as exc:
            logger.error("search_documents retrieval error: %s", exc)
            return f"Document search failed: {exc}"

        if not child_hits:
            return (
                "No relevant content found in uploaded documents for this query. "
                "The documents may not contain information about this topic, or "
                "try rephrasing the query with different terminology."
            )

        parent_ids = list({hit["parent_id"] for hit in child_hits if hit.get("parent_id")})
        try:
            parents = await retrieval_repo.fetch_parent_contexts(parent_ids) if parent_ids else []
        except Exception as exc:
            logger.warning("search_documents parent fetch failed: %s", exc)
            parents = []

        parent_map = {p["id"]: p["parent_chunk_content"] for p in parents}

        lines: list[str] = [f"Found {len(child_hits)} relevant passages:\n"]
        for i, hit in enumerate(child_hits, 1):
            filename = hit.get("filename", "unknown")
            similarity = hit.get("similarity", 0.0)
            parent_content = parent_map.get(hit.get("parent_id", ""), "")
            content = parent_content or hit.get("chunk_content", "")

            lines.append(f"[{i}] Source: {filename} (relevance: {similarity:.2f})")
            lines.append(content.strip())
            lines.append("")

        return "\n".join(lines)

    @tool
    def get_uploaded_documents() -> str:
        """List all documents that have been uploaded and indexed in the current session.

        Use this at the start of a conversation to understand what documents are available,
        or when the user asks 'what papers do you have?' or 'what have I uploaded?'

        Also use this before calling search_documents if you are unsure whether a
        relevant document exists in the session.

        Returns:
            A list of uploaded documents with filenames, descriptions, and chunk counts.
            Returns a message if no documents have been uploaded yet.
        """
        logger.info("get_uploaded_documents: session=%s", session_id)

        try:
            session = memory_repo.get_session(session_id=session_id, user_id=user_id)
            if session is None:
                return "Session not found."
        except Exception as exc:
            logger.error("get_uploaded_documents session lookup failed: %s", exc)
            return f"Could not retrieve session information: {exc}"

        try:
            docs = memory_repo.get_session_documents(
                session_id=session_id,
                user_id=user_id,
            )
        except AttributeError:
            return (
                "Document listing is not available in this environment. "
                "Use search_documents to search directly — if documents have been "
                "uploaded they will appear in search results."
            )
        except Exception as exc:
            logger.error("get_uploaded_documents error: %s", exc)
            return f"Could not retrieve document list: {exc}"

        if not docs:
            return (
                "No documents have been uploaded to this session yet. "
                "Ask the user to upload a PDF or DOCX file to get started."
            )

        lines = [f"Documents in this session ({len(docs)} total):\n"]
        for i, doc in enumerate(docs, 1):
            description = doc.get("file_description") or "No description provided"
            parent_chunks = doc.get("parent_chunks", "?")
            child_chunks = doc.get("child_chunks", "?")
            ingested_at = doc.get("ingested_at", "unknown")
            lines.append(
                f"[{i}] {doc['filename']}\n"
                f"     Description: {description}\n"
                f"     Chunks: {parent_chunks} parent / {child_chunks} child\n"
                f"     Indexed: {ingested_at}"
            )
        return "\n".join(lines)

    @tool
    def extract_paper_metadata(filename: str) -> str:
        """Extract structured metadata from an uploaded research paper.

        Returns the title, authors, abstract, publication year, and section headings
        identified in the document during indexing.

        Use this when:
        - The user asks who wrote a paper.
        - The user asks what year a paper was published.
        - You need to cite a paper properly.
        - You want an overview of the paper's structure before searching within it.

        Args:
            filename: The exact filename of the uploaded document, as returned by
                      get_uploaded_documents. Example: "attention_is_all_you_need.pdf"

        Returns:
            Structured metadata for the paper, or an explanation if not found.
        """
        logger.info("extract_paper_metadata: filename=%r session=%s", filename, session_id)

        try:
            chunks = memory_repo.get_chunks_by_filename(
                session_id=session_id,
                user_id=user_id,
                filename=filename,
                content_types=["text"],
                limit=20,
            )
        except AttributeError:
            return (
                "Metadata extraction via the repository is not yet implemented. "
                "Use search_documents with a query like 'title authors abstract' "
                f"to find metadata for {filename!r}."
            )
        except Exception as exc:
            logger.error("extract_paper_metadata error: %s", exc)
            return f"Could not retrieve metadata for {filename!r}: {exc}"

        if not chunks:
            return (
                f"No content found for {filename!r}. "
                "Verify the filename using get_uploaded_documents."
            )

        metadata_fields: dict[str, str] = {}
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}

            for field in ("title", "authors", "abstract", "year", "doi"):
                if field not in metadata_fields and meta.get(field):
                    metadata_fields[field] = str(meta[field])

        lines = [f"Metadata for {filename}:"]
        if metadata_fields:
            for key, value in metadata_fields.items():
                lines.append(f"  {key.capitalize()}: {value}")
        else:
            lines.append(
                "  No structured metadata was extracted during indexing. "
                "Try search_documents with query 'title authors abstract year' "
                "to locate this information in the document text."
            )

        return "\n".join(lines)

    @tool
    async def summarize_document(filename: str, focus: str = "") -> str:
        """Generate a summary of a specific uploaded document.

        Use this when the user asks for an overview of a paper, or when you need
        to understand the overall content of a document before answering a detailed
        question about it.

        For targeted questions about specific sections, use search_documents instead —
        it is faster and more precise. Use summarize_document only when a broad
        overview is what is needed.

        Args:
            filename: The exact filename of the document to summarize, as returned by
                      get_uploaded_documents.
            focus: Optional aspect to focus the summary on. Examples: "methodology",
                   "results", "limitations", "contributions". Leave empty for a general
                   summary.

        Returns:
            A summary of the document's key content, optionally focused on a specific aspect.
        """
        logger.info(
            "summarize_document: filename=%r focus=%r session=%s", filename, focus, session_id
        )

        focus_query = focus if focus else "overview introduction abstract conclusion contributions"

        try:
            query_vec = embedder.embed_query(
                f"{focus_query} {filename}" if focus else focus_query
            )
            hits = await retrieval_repo.search(
                query_embedding=query_vec,
                top_k=8,
                session_id=session_id,
            )
        except Exception as exc:
            logger.error("summarize_document retrieval error: %s", exc)
            return f"Could not retrieve content for {filename!r}: {exc}"

        file_hits = [h for h in hits if h.get("filename") == filename]

        if not file_hits:
            return (
                f"No content found for {filename!r}. "
                "Check the filename with get_uploaded_documents."
            )

        parent_ids = list({h["parent_id"] for h in file_hits if h.get("parent_id")})
        try:
            parents = await retrieval_repo.fetch_parent_contexts(parent_ids) if parent_ids else []
        except Exception as exc:
            logger.warning("summarize_document parent fetch failed: %s", exc)
            parents = []

        parent_map = {p["id"]: p["parent_chunk_content"] for p in parents}

        passages = []
        for hit in file_hits:
            content = parent_map.get(hit.get("parent_id", ""), "") or hit.get("chunk_content", "")
            if content.strip():
                passages.append(content.strip())

        context = "\n\n".join(passages[:6])
        focus_note = f" focusing on {focus!r}" if focus else ""

        return (
            f"Content retrieved from {filename!r}{focus_note} "
            f"({len(passages)} passages):\n\n{context}\n\n"
            "[Use this content to generate a summary. Do not return raw passages "
            "to the user — synthesize them into a coherent response.]"
        )

    return [
        search_documents,
        get_uploaded_documents,
        extract_paper_metadata,
        summarize_document,
    ]
