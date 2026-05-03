"""Supervisor agent that delegates to specialist worker agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool

from ..chat_service import ChatService
from ..core.models import ChatRecord
from ._shared import extract_agent_run_result
from .workers import (
    ComputationWorkerAgent,
    DataAnalysisWorkerAgent,
    DocumentExtractionWorkerAgent,
    DocumentResearchWorkerAgent,
    WebResearchWorkerAgent,
)

_SUPERVISOR_PROMPT = """You are the Supervisor Orchestration Agent.

Your job is to decide whether the user's request needs:
- uploaded document research
- web research
- exact calculation
- data analysis (pandas/Python code in a sandbox)
- document entity extraction and quality assessment
- or a combination of these

You do not directly use low-level retrieval tools. Instead, delegate with the
specialist worker tools available to you:
- ask_document_worker       — search and summarize uploaded session documents
- ask_web_worker            — search the live web or fetch a webpage
- ask_computation_worker    — exact arithmetic or formula evaluation
- ask_data_analysis_worker  — run Python/pandas analysis on data in a sandbox
- ask_document_extraction_worker — extract structured entities from a document
                                   using RaV-IDP (with fidelity scoring)

Rules:
- Delegate only when needed.
- Use the document worker for uploaded-file questions.
- Use the web worker for current or external information.
- Use the computation worker for exact numerical work.
- Use the data analysis worker when the user asks about trends, statistics,
  distributions, or wants code run against a dataset.
- Use the document extraction worker when the user wants entities extracted
  from a document, or wants to know the document's extraction quality/fidelity.
- You may call multiple workers when the task needs multi-source synthesis.
- Avoid redundant delegations.
- Synthesize the worker outputs into one clear final answer for the user.
- If evidence is incomplete, say what is known and what remains uncertain.
- Do not expose raw chain-of-thought.
"""


@dataclass
class SupervisorAgentResult:
    response: str
    tools_used: list[str]
    step_count: int


class SupervisorOrchestrationAgent:
    def __init__(
        self,
        chat_service: ChatService,
        document_worker: DocumentResearchWorkerAgent,
        web_worker: WebResearchWorkerAgent,
        computation_worker: ComputationWorkerAgent,
        data_analysis_worker: DataAnalysisWorkerAgent,
        document_extraction_worker: DocumentExtractionWorkerAgent,
        short_term_history: list[ChatRecord] | None = None,
        long_term_history: list[dict] | None = None,
    ) -> None:
        self._chat_service = chat_service
        self._document_worker = document_worker
        self._web_worker = web_worker
        self._computation_worker = computation_worker
        self._data_analysis_worker = data_analysis_worker
        self._document_extraction_worker = document_extraction_worker
        self._short_term_history = short_term_history or []
        self._long_term_history = long_term_history or []

    def _build_system_prompt(self) -> str:
        base = self._chat_service._build_system_prompt(  # noqa: SLF001
            short_term_history=self._short_term_history,
            long_term_history=self._long_term_history,
            rag_context=None,
        )
        return f"{base}\n\n{_SUPERVISOR_PROMPT}"

    def _build_worker_tools(self, worker_usage: list[str]):
        document_worker = self._document_worker
        web_worker = self._web_worker
        computation_worker = self._computation_worker
        data_analysis_worker = self._data_analysis_worker
        document_extraction_worker = self._document_extraction_worker

        @tool
        async def ask_document_worker(task: str) -> str:
            """Delegate uploaded-document research to the document worker."""
            result = await document_worker.arun(task)
            worker_usage.append("document_worker")
            worker_usage.extend([f"document_worker:{name}" for name in result.tools_used])
            return f"Document Worker Findings:\n{result.response}"

        @tool
        async def ask_web_worker(task: str) -> str:
            """Delegate external or current-information research to the web worker."""
            result = await web_worker.arun(task)
            worker_usage.append("web_worker")
            worker_usage.extend([f"web_worker:{name}" for name in result.tools_used])
            return f"Web Worker Findings:\n{result.response}"

        @tool
        async def ask_computation_worker(task: str) -> str:
            """Delegate exact numerical reasoning or arithmetic to the computation worker."""
            result = await computation_worker.arun(task)
            worker_usage.append("computation_worker")
            worker_usage.extend([f"computation_worker:{name}" for name in result.tools_used])
            return f"Computation Worker Findings:\n{result.response}"

        @tool
        async def ask_data_analysis_worker(task: str) -> str:
            """Delegate data analysis, statistics, or pandas/Python code execution to the data analysis worker.

            Pass any CSV data or numeric content as part of the task description so the worker
            can include it in the sandbox execution.
            """
            result = await data_analysis_worker.arun(task)
            worker_usage.append("data_analysis_worker")
            worker_usage.extend([f"data_analysis_worker:{name}" for name in result.tools_used])
            return f"Data Analysis Worker Findings:\n{result.response}"

        @tool
        async def ask_document_extraction_worker(task: str) -> str:
            """Delegate document entity extraction or fidelity assessment to the document extraction worker.

            Include the document path in the task description. The worker will run
            RaV-IDP to extract structured entities with quality/fidelity scoring.
            """
            result = await document_extraction_worker.arun(task)
            worker_usage.append("document_extraction_worker")
            worker_usage.extend([f"document_extraction_worker:{name}" for name in result.tools_used])
            return f"Document Extraction Worker Findings:\n{result.response}"

        return [
            ask_document_worker,
            ask_web_worker,
            ask_computation_worker,
            ask_data_analysis_worker,
            ask_document_extraction_worker,
        ]

    async def arun(self, user_message: str) -> SupervisorAgentResult:
        worker_usage: list[str] = []
        graph = create_agent(
            model=self._chat_service.llm_provider.llm,
            tools=self._build_worker_tools(worker_usage),
            system_prompt=self._build_system_prompt(),
            name="supervisor_orchestration_agent",
        )
        result = await graph.ainvoke({"messages": [{"role": "user", "content": user_message}]})
        extracted = extract_agent_run_result(
            result,
            "The supervisor agent could not produce a final response.",
        )
        tools_used = extracted.tools_used + worker_usage
        return SupervisorAgentResult(
            response=extracted.response,
            tools_used=tools_used,
            step_count=max(len(tools_used), 1),
        )
