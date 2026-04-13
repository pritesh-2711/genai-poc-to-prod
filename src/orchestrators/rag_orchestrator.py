"""Top-level RAG orchestrator — routes to fast or deep subgraph.

Graph topology:
    START
      → [mode router]
          → "fast": fast_subgraph
          → "deep": deep_subgraph
      → END

The router graph is compiled with a MemorySaver checkpointer so that
interrupt() in the deep subgraph persists state across HTTP requests.

Usage (from the API):
    # New turn
    result = await orchestrator.ainvoke(initial_state, thread_id=str(session_id))

    # Check if paused for clarification
    if orchestrator.is_interrupted(thread_id):
        # Return clarification_question to user as assistant reply

    # Resume interrupted turn (next request)
    result = await orchestrator.aresume(thread_id, user_clarification)

Thread-ID strategy:
    thread_id = str(user_chat_id)  — unique per chat turn.
    The API tracks which thread_id is awaiting clarification in
    app.state.pending_clarifications: dict[session_id_str, thread_id_str].
"""

import logging
from typing import AsyncIterator, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from ..chat_service import ChatService
from ..core.models import ChatConfig, RerankerConfig
from ..databases.retrieval import PgVectorRetrievalRepository
from ..embedding.base import BaseEmbedder
from ..memory.repository import MemoryRepository
from ..reranker.base import BaseReranker
from .deep_orchestrator import DeepOrchestrator
from .fast_orchestrator import FastOrchestrator
from .state import RAGState

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """Top-level orchestrator that routes requests to fast or deep subgraphs.

    This is NOT a BaseOrchestrator subclass — it is a composition layer that
    holds compiled fast and deep subgraphs and wraps them in a router graph.

    Args:
        embedder:        Text embedder singleton (shared with API layer).
        retrieval_repo:  Async pgvector retrieval repository.
        reranker:        Cross-encoder reranker.
        chat_service:    LLM provider wrapper.
        memory_repo:     Sync PostgreSQL memory repository.
        reranker_config: Reranker settings.
        chat_config:     Chat settings.
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
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> None:
        shared_kwargs = dict(
            embedder=embedder,
            retrieval_repo=retrieval_repo,
            reranker=reranker,
            chat_service=chat_service,
            memory_repo=memory_repo,
            reranker_config=reranker_config,
            chat_config=chat_config,
        )

        self._fast = FastOrchestrator(**shared_kwargs)
        self._deep = DeepOrchestrator(**shared_kwargs)

        # Checkpointer is injected by main.py so the deployment backend
        # (MemorySaver for local, AsyncRedisSaver for cloud) is decided at startup.
        self._checkpointer = checkpointer if checkpointer is not None else MemorySaver()
        self._graph = self._build_router_graph()

        logger.info("RAGOrchestrator initialised (fast + deep subgraphs compiled).")

    def _build_router_graph(self) -> CompiledStateGraph:
        """Wire fast and deep subgraphs behind a mode-based conditional edge."""
        fast_graph = self._fast.build_graph()
        deep_graph = self._deep.build_graph()

        builder = StateGraph(RAGState)
        builder.add_node("fast_graph", fast_graph)
        builder.add_node("deep_graph", deep_graph)

        builder.add_conditional_edges(
            START,
            lambda state: state.get("mode", "fast"),
            {"fast": "fast_graph", "deep": "deep_graph"},
        )
        builder.add_edge("fast_graph", END)
        builder.add_edge("deep_graph", END)

        return builder.compile(checkpointer=self._checkpointer)

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    async def ainvoke(self, initial_state: RAGState, thread_id: str) -> RAGState:
        """Start a new graph run and return the final state.

        Args:
            initial_state: Input fields (original_query, query_embedding, mode,
                           session_id, user_id, user_chat_id).
            thread_id:     Unique ID for this graph run (use str(user_chat_id)).

        Returns:
            Final RAGState after the graph completes (or is interrupted).
        """
        config = {"configurable": {"thread_id": thread_id}}
        result = await self._graph.ainvoke(initial_state, config=config)
        return result

    async def aresume(self, thread_id: str, user_clarification: str) -> RAGState:
        """Resume an interrupted graph (after query_clarification interrupt).

        Args:
            thread_id:          The same thread_id used in the original ainvoke call.
            user_clarification: The user's reply to the clarification question.

        Returns:
            Final RAGState after the graph completes.
        """
        config = {"configurable": {"thread_id": thread_id}}
        result = await self._graph.ainvoke(
            Command(resume=user_clarification),
            config=config,
        )
        return result

    async def astream_updates(
        self,
        initial_state: RAGState,
        thread_id: str,
    ) -> AsyncIterator[tuple]:
        """Stream per-node state updates for status reporting.

        Yields tuples of ``(namespace_tuple, {node_name: state_delta})``.
        Use ``namespace_tuple`` to distinguish inner subgraph nodes (non-empty)
        from outer graph completions (empty tuple).

        Args:
            initial_state: Same as ainvoke.
            thread_id:     Same as ainvoke.
        """
        config = {"configurable": {"thread_id": thread_id}}
        async for chunk in self._graph.astream(
            initial_state,
            config=config,
            stream_mode="updates",
            subgraphs=True,
        ):
            yield chunk

    def get_graph_state(self, thread_id: str):
        """Return the state snapshot for a thread (after astream or ainvoke).

        Use to extract final values or check for interrupts.
        """
        config = {"configurable": {"thread_id": thread_id}}
        return self._graph.get_state(config)

    def is_interrupted(self, thread_id: str) -> bool:
        """Return True if the graph for thread_id is currently paused by interrupt().

        Args:
            thread_id: The thread to inspect.

        Returns:
            True if the thread is awaiting a resume() call.
        """
        config = {"configurable": {"thread_id": thread_id}}
        state = self._graph.get_state(config)
        if state is None:
            return False
        return bool(state.next)  # non-empty `next` means the graph is interrupted

    def get_clarification_question(self, thread_id: str) -> Optional[str]:
        """Return the clarification question from an interrupted thread.

        Args:
            thread_id: The interrupted thread ID.

        Returns:
            The clarification question string, or None if not interrupted.
        """
        config = {"configurable": {"thread_id": thread_id}}
        state = self._graph.get_state(config)
        if state is None or not state.next:
            return None
        return state.values.get("clarification_question")
