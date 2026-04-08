"""Abstract base orchestrator — shared node implementations.

All node methods live here so FastOrchestrator and DeepOrchestrator can
reuse them without duplication.  Subclasses only need to implement
``build_graph()`` to wire the nodes into a LangGraph subgraph.

Bottom-to-top design:
  1. Individual node methods (this file)      ← leaves
  2. FastOrchestrator / DeepOrchestrator      ← subgraphs
  3. RAGOrchestrator                          ← top-level router
"""

import logging
from abc import ABC, abstractmethod
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph

from ..chat_service import ChatService
from ..core.models import ChatConfig, RerankerConfig
from ..databases.retrieval import PgVectorRetrievalRepository
from ..embedding.base import BaseEmbedder
from ..memory.repository import MemoryRepository
from ..reranker.base import BaseReranker
from .state import RAGState, SubQueryState

logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS = 8_000
_RAG_TOP_K = 10
_LONG_TERM_TOP_K = 10


class BaseOrchestrator(ABC):
    """Abstract base for all LangGraph RAG orchestrators.

    Holds references to every infrastructure dependency and exposes async node
    methods that subclasses compose into their specific subgraphs.

    Args:
        embedder:        Provider-agnostic text embedder (same instance as API).
        retrieval_repo:  Async pgvector retrieval repository.
        reranker:        Cross-encoder reranker.
        chat_service:    LLM provider wrapper (generation + system-prompt builder).
        memory_repo:     Sync PostgreSQL memory repository (chat history).
        reranker_config: Reranker settings (top_k, enabled flag).
        chat_config:     Chat settings (short_term_limit, similarity threshold).
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        retrieval_repo: PgVectorRetrievalRepository,
        reranker: BaseReranker,
        chat_service: ChatService,
        memory_repo: MemoryRepository,
        reranker_config: RerankerConfig,
        chat_config: ChatConfig,
    ) -> None:
        self._embedder = embedder
        self._retrieval_repo = retrieval_repo
        self._reranker = reranker
        self._chat_service = chat_service
        self._memory_repo = memory_repo
        self._reranker_config = reranker_config
        self._chat_config = chat_config

    @abstractmethod
    def build_graph(self) -> CompiledStateGraph:
        """Compile and return the subgraph for this orchestrator."""

    # ──────────────────────────────────────────────────────────────────────────
    # Shared node: resolve_memory
    # Reads short-term + long-term history from PostgreSQL.
    # Memory is used ONLY for generation (system prompt injection) — NOT retrieval.
    # ──────────────────────────────────────────────────────────────────────────

    async def _resolve_memory_node(self, state: RAGState) -> dict:
        """Fetch short-term and long-term conversation memory."""
        session_id = UUID(state["session_id"])
        query_vec = state["query_embedding"]
        user_chat_id = state.get("user_chat_id", "")
        short_term_limit = self._chat_config.short_term_limit
        similarity_threshold = self._chat_config.long_term_similarity_threshold

        raw_history = self._memory_repo.get_conversation_history(
            session_id=session_id, limit=short_term_limit + 1
        )
        short_term_history = raw_history[:-1]  # exclude the just-added user message

        session_exceeds_limit = len(raw_history) == short_term_limit + 1
        if not session_exceeds_limit:
            return {
                "short_term_history": short_term_history,
                "long_term_history": [],
            }

        long_term_raw = await self._retrieval_repo.search_conversation_history(
            query_embedding=query_vec,
            session_id=session_id,
            top_k=_LONG_TERM_TOP_K,
            exclude_chat_id=user_chat_id,
        )

        long_term_above_threshold = [
            r for r in long_term_raw if r["similarity"] >= similarity_threshold
        ]
        short_term_ids = {str(r.chat_id) for r in short_term_history}
        long_term_history = [
            r for r in long_term_above_threshold if r["chat_id"] not in short_term_ids
        ]

        logger.debug(
            f"[memory] session={session_id} | short={len(short_term_history)} "
            f"long={len(long_term_history)}"
        )
        return {
            "short_term_history": short_term_history,
            "long_term_history": long_term_history,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Shared node: retrieve
    # Single vector search for fast / low-complexity / moderate paths.
    # For high-complexity (sub-queries), see _retrieve_sub_query_node below.
    # ──────────────────────────────────────────────────────────────────────────

    async def _retrieve_node(self, state: RAGState) -> dict:
        """Vector search using query_embedding (or retrieval_embedding for rewritten queries)."""
        session_id = UUID(state["session_id"])

        # Use reframed-query embedding if query was rewritten, else original
        retrieval_query = state.get("retrieval_query") or state.get("original_query", "")

        # Re-embed only if the retrieval query differs from the original query
        if retrieval_query != state.get("original_query", ""):
            query_vec = self._embedder.embed_query(retrieval_query)
        else:
            query_vec = state["query_embedding"]

        try:
            chunks = await self._retrieval_repo.search(
                query_embedding=query_vec,
                top_k=_RAG_TOP_K,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning(f"[retrieve] vector search failed: {exc}")
            chunks = []

        return {"raw_chunks": chunks}

    # ──────────────────────────────────────────────────────────────────────────
    # Shared node: retrieve_sub_query
    # Called in parallel via Send API for high-complexity deep mode.
    # Receives SubQueryState; returns raw_chunks which accumulate via operator.add.
    # ──────────────────────────────────────────────────────────────────────────

    async def _retrieve_sub_query_node(self, state: SubQueryState) -> dict:
        """Embed and search a single sub-query (runs in parallel via Send API)."""
        session_id = UUID(state["session_id"])
        sub_query = state["sub_query"]

        query_vec = self._embedder.embed_query(sub_query)

        try:
            chunks = await self._retrieval_repo.search(
                query_embedding=query_vec,
                top_k=_RAG_TOP_K,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning(f"[retrieve_sub_query] search failed for '{sub_query}': {exc}")
            chunks = []

        return {"raw_chunks": chunks}

    # ──────────────────────────────────────────────────────────────────────────
    # Shared node: rerank_and_build_context
    # Runs the cross-encoder on raw_chunks, fetches parent contexts + co-located
    # chunks, then formats the rag_context string.
    # ──────────────────────────────────────────────────────────────────────────

    async def _rerank_and_build_context_node(self, state: RAGState) -> dict:
        """Rerank raw chunks and build the RAG context string."""
        raw_chunks: list[dict] = state.get("raw_chunks") or []

        if not raw_chunks:
            return {"reranked_chunks": [], "rag_context": ""}

        # ── 1. Cross-encoder reranking ────────────────────────────────────────
        query = state.get("original_query", "")
        top_k = self._reranker_config.top_k if self._reranker_config.enabled else len(raw_chunks)

        if self._reranker_config.enabled:
            reranked = self._reranker.rerank(query=query, chunks=raw_chunks, top_k=top_k)
        else:
            reranked = raw_chunks[:top_k]

        if not reranked:
            return {"reranked_chunks": [], "rag_context": ""}

        # ── 2. Fetch parent contexts for top-K reranked chunks ────────────────
        session_id = UUID(state["session_id"])
        parent_ids = list({
            c["parent_id"] for c in reranked if c.get("parent_id")
        })

        if parent_ids:
            try:
                parents = await self._retrieval_repo.fetch_parent_contexts(parent_ids)
            except Exception as exc:
                logger.warning(f"[rerank] fetch_parent_contexts failed: {exc}")
                parents = []
        else:
            parents = []

        # ── 3. Co-located table/image chunks from same pages ──────────────────
        colocated: list[dict] = []
        if parents:
            pages_seen: set[int] = set()
            filenames_seen: set[str] = set()
            for p in parents:
                meta = p.get("metadata") or {}
                page = meta.get("page")
                if isinstance(page, int):
                    pages_seen.add(page)
                if p.get("filename"):
                    filenames_seen.add(p["filename"])

            if pages_seen and filenames_seen:
                try:
                    colocated = await self._retrieval_repo.fetch_colocated_chunks(
                        session_id=session_id,
                        pages=list(pages_seen),
                        filenames=list(filenames_seen),
                    )
                except Exception as exc:
                    logger.warning(f"[rerank] fetch_colocated_chunks failed: {exc}")

        # ── 4. Build context string ───────────────────────────────────────────
        context_parts: list[str] = []
        total = 0

        passages = [p["parent_chunk_content"] for p in parents if p.get("parent_chunk_content")]
        if not passages:
            passages = [c["chunk_content"] for c in reranked if c.get("chunk_content")]

        for i, passage in enumerate(passages, 1):
            snippet = passage.strip()
            if total + len(snippet) > _MAX_CONTEXT_CHARS:
                snippet = snippet[: _MAX_CONTEXT_CHARS - total]
            context_parts.append(f"[{i}] {snippet}")
            total += len(snippet)
            if total >= _MAX_CONTEXT_CHARS:
                break

        seen_content: set[str] = set()
        for chunk in colocated:
            if total >= _MAX_CONTEXT_CHARS:
                break
            text = chunk.get("chunk_content", "").strip()
            if not text or text in seen_content:
                continue
            seen_content.add(text)
            ctype = chunk.get("content_type", "table")
            page = (chunk.get("metadata") or {}).get("page", "?")
            snippet = f"[{ctype.capitalize()} p.{page}] {text}"
            if total + len(snippet) > _MAX_CONTEXT_CHARS:
                snippet = snippet[: _MAX_CONTEXT_CHARS - total]
            context_parts.append(snippet)
            total += len(snippet)

        rag_context = "\n\n".join(context_parts)
        return {"reranked_chunks": reranked, "rag_context": rag_context}

    # ──────────────────────────────────────────────────────────────────────────
    # Shared node: generate
    # Calls ChatService with rag_context + memory injected into system prompt.
    # In deep mode, includes correction_note on retry iterations.
    # ──────────────────────────────────────────────────────────────────────────

    async def _generate_node(self, state: RAGState) -> dict:
        """Generate an LLM response grounded by rag_context and memory."""
        user_message = state.get("original_query", "")
        rag_context = state.get("rag_context") or None
        short_term_history = state.get("short_term_history") or []
        long_term_history = state.get("long_term_history") or []
        correction_note = state.get("correction_note", "")

        # Prepend correction guidance when retrying after a failed validation
        effective_message = user_message
        if correction_note:
            effective_message = (
                f"{user_message}\n\n"
                f"[Correction guidance for retry: {correction_note}]"
            )

        response = await self._chat_service.get_response_async(
            user_message=effective_message,
            short_term_history=short_term_history,
            long_term_history=long_term_history,
            rag_context=rag_context,
        )

        iteration = state.get("iteration_count", 0)
        logger.debug(f"[generate] iteration={iteration} | response_len={len(response)}")
        return {"llm_response": response}
