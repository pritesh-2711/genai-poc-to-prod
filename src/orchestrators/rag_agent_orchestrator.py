"""Single-agent RAG orchestrator.

This keeps deterministic memory resolution from BaseOrchestrator, then hands off
reasoning/tool selection/synthesis to a single high-level agent.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..agents import SingleRAGAgent
from ..tools import build_document_tools, calculate, fetch_webpage, web_search
from .base import BaseOrchestrator
from .state import RAGState

logger = logging.getLogger(__name__)


class RAGAgentOrchestrator(BaseOrchestrator):
    """Compiles the single-agent RAG subgraph."""

    async def _run_agent_node(self, state: RAGState) -> dict:
        tools = build_document_tools(
            embedder=self._embedder,
            retrieval_repo=self._retrieval_repo,
            memory_repo=self._memory_repo,
            session_id=state["session_id"],
            user_id=state["user_id"],
        ) + [web_search, fetch_webpage, calculate]

        agent = SingleRAGAgent(
            chat_service=self._chat_service,
            tools=tools,
            short_term_history=state.get("short_term_history") or [],
            long_term_history=state.get("long_term_history") or [],
        )
        result = await agent.arun(state.get("original_query", ""))
        return {
            "final_response": result.response,
            "tools_used": result.tools_used,
            "agent_step_count": result.step_count,
        }

    def build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(RAGState)
        builder.add_node("resolve_memory", self._resolve_memory_node)
        builder.add_node("run_agent", self._run_agent_node)
        builder.add_edge(START, "resolve_memory")
        builder.add_edge("resolve_memory", "run_agent")
        builder.add_edge("run_agent", END)
        graph = builder.compile()
        logger.info("RAGAgentOrchestrator subgraph compiled.")
        return graph
