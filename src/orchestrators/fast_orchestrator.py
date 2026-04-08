"""Fast-mode subgraph orchestrator.

Graph topology:
    START
      → resolve_memory     (fetch short-term + long-term history from DB)
      → retrieve           (single vector search using query_embedding)
      → rerank_and_build_context  (cross-encoder + parent fetch + context formatting)
      → generate           (LLM response grounded by rag_context + memory)
      → END

Characteristics:
  - No LLM calls before retrieval (no intent analysis, no query rewriting)
  - No self-validation or correction loop
  - 1 embed (pre-computed by API) + 1 DB query + 1 rerank + 1 LLM call
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .base import BaseOrchestrator
from .state import RAGState

logger = logging.getLogger(__name__)


class FastOrchestrator(BaseOrchestrator):
    """Compiles the fast-mode RAG subgraph.

    All node implementations are inherited from BaseOrchestrator.
    This class only wires them together in the correct order.
    """

    def build_graph(self) -> CompiledStateGraph:
        """Build and compile the fast subgraph.

        Returns:
            A compiled LangGraph subgraph ready to be embedded in the
            top-level router or invoked directly.
        """
        builder = StateGraph(RAGState)

        builder.add_node("resolve_memory", self._resolve_memory_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("rerank_and_build_context", self._rerank_and_build_context_node)
        builder.add_node("generate", self._generate_node)

        builder.add_edge(START, "resolve_memory")
        builder.add_edge("resolve_memory", "retrieve")
        builder.add_edge("retrieve", "rerank_and_build_context")
        builder.add_edge("rerank_and_build_context", "generate")
        builder.add_edge("generate", END)

        graph = builder.compile()
        logger.info("FastOrchestrator subgraph compiled.")
        return graph
