"""Supervisor-agent RAG orchestrator.

Keeps memory deterministic, then delegates to a supervisor agent that can call
specialized document, web, computation, data analysis, and document extraction workers.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..agents import SupervisorOrchestrationAgent
from ..agents.workers import (
    ComputationWorkerAgent,
    DataAnalysisWorkerAgent,
    DocumentExtractionWorkerAgent,
    DocumentResearchWorkerAgent,
    WebResearchWorkerAgent,
)
from ..tools import build_document_tools
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

        # Load stateless tools from the MCP server.
        # Falls back to empty list if MCP is unavailable — workers handle gracefully.
        mcp = self._mcp_tool_loader
        web_tools = mcp.get_tools(["web_search", "fetch_webpage"]) if mcp else []
        computation_tools = mcp.get_tools(["calculate"]) if mcp else []
        analysis_tools = mcp.get_tools(["analyse"]) if mcp else []
        extraction_tools = mcp.get_tools([
            "rav_idp_process_and_ingest",
            "rav_idp_get_document_fidelity",
        ]) if mcp else []

        worker_kwargs = dict(
            chat_service=self._chat_service,
            short_term_history=short_term_history,
            long_term_history=long_term_history,
        )

        document_worker = DocumentResearchWorkerAgent(tools=document_tools, **worker_kwargs)
        web_worker = WebResearchWorkerAgent(tools=web_tools, **worker_kwargs)
        computation_worker = ComputationWorkerAgent(tools=computation_tools, **worker_kwargs)
        data_analysis_worker = DataAnalysisWorkerAgent(tools=analysis_tools, **worker_kwargs)
        document_extraction_worker = DocumentExtractionWorkerAgent(
            tools=extraction_tools, **worker_kwargs
        )

        supervisor = SupervisorOrchestrationAgent(
            chat_service=self._chat_service,
            document_worker=document_worker,
            web_worker=web_worker,
            computation_worker=computation_worker,
            data_analysis_worker=data_analysis_worker,
            document_extraction_worker=document_extraction_worker,
            short_term_history=short_term_history,
            long_term_history=long_term_history,
        )
        result = await supervisor.arun(state.get("original_query", ""))
        return {
            "final_response": result.response,
            "tools_used": result.tools_used,
            "agent_step_count": result.step_count,
            "charts": result.charts,
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
