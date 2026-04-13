"""Supervisor-agent RAG orchestrator.

Keeps memory deterministic, then delegates to a supervisor agent that can call
specialized document, web, and computation workers.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..agents import SupervisorOrchestrationAgent
from ..agents.workers import (
    ComputationWorkerAgent,
    DocumentResearchWorkerAgent,
    WebResearchWorkerAgent,
)
from ..tools import build_document_tools, calculate, fetch_webpage, web_search
from .base import BaseOrchestrator
from .state import RAGState

logger = logging.getLogger(__name__)


class SupervisorAgentOrchestrator(BaseOrchestrator):
    """Compiles the supervisor multi-agent RAG subgraph."""

    async def _run_supervisor_node(self, state: RAGState) -> dict:
        short_term_history = state.get("short_term_history") or []
        long_term_history = state.get("long_term_history") or []

        document_tools = build_document_tools(
            embedder=self._embedder,
            retrieval_repo=self._retrieval_repo,
            memory_repo=self._memory_repo,
            session_id=state["session_id"],
            user_id=state["user_id"],
        )
        web_tools = [web_search, fetch_webpage]
        computation_tools = [calculate]

        document_worker = DocumentResearchWorkerAgent(
            chat_service=self._chat_service,
            tools=document_tools,
            short_term_history=short_term_history,
            long_term_history=long_term_history,
        )
        web_worker = WebResearchWorkerAgent(
            chat_service=self._chat_service,
            tools=web_tools,
            short_term_history=short_term_history,
            long_term_history=long_term_history,
        )
        computation_worker = ComputationWorkerAgent(
            chat_service=self._chat_service,
            tools=computation_tools,
            short_term_history=short_term_history,
            long_term_history=long_term_history,
        )

        supervisor = SupervisorOrchestrationAgent(
            chat_service=self._chat_service,
            document_worker=document_worker,
            web_worker=web_worker,
            computation_worker=computation_worker,
            short_term_history=short_term_history,
            long_term_history=long_term_history,
        )
        result = await supervisor.arun(state.get("original_query", ""))
        return {
            "final_response": result.response,
            "tools_used": result.tools_used,
            "agent_step_count": result.step_count,
        }

    def build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(RAGState)
        builder.add_node("resolve_memory", self._resolve_memory_node)
        builder.add_node("run_supervisor", self._run_supervisor_node)
        builder.add_edge(START, "resolve_memory")
        builder.add_edge("resolve_memory", "run_supervisor")
        builder.add_edge("run_supervisor", END)
        graph = builder.compile()
        logger.info("SupervisorAgentOrchestrator subgraph compiled.")
        return graph
