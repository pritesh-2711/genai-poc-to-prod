"""Worker agent specialized in uploaded-document research."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from ...chat_service import ChatService
from ...core.models import ChatRecord
from .._shared import AgentRunResult, extract_agent_run_result

_DOCUMENT_RESEARCH_PROMPT = """You are the Document Research Worker.

Role:
- Answer only using the uploaded session documents and their document tools.
- Prefer precise, grounded findings over broad generalizations.
- If the uploaded documents do not contain enough evidence, say that clearly.
- Do not use external-web assumptions.

Working style:
- Start by checking what documents are available when helpful.
- Search documents with focused sub-queries.
- Summarize or extract paper metadata only when those tools are the best fit.
- Return concise, evidence-grounded findings for the supervisor to synthesize.
"""


class DocumentResearchWorkerAgent:
    def __init__(
        self,
        chat_service: ChatService,
        tools: list[Any],
        short_term_history: list[ChatRecord] | None = None,
        long_term_history: list[dict] | None = None,
    ) -> None:
        self._chat_service = chat_service
        self._tools = tools
        self._short_term_history = short_term_history or []
        self._long_term_history = long_term_history or []

    def _build_system_prompt(self) -> str:
        base = self._chat_service._build_system_prompt(  # noqa: SLF001
            short_term_history=self._short_term_history,
            long_term_history=self._long_term_history,
            rag_context=None,
        )
        return f"{base}\n\n{_DOCUMENT_RESEARCH_PROMPT}"

    async def arun(self, task: str) -> AgentRunResult:
        graph = create_agent(
            model=self._chat_service.llm_provider.llm,
            tools=self._tools,
            system_prompt=self._build_system_prompt(),
            name="document_research_worker",
        )
        result = await graph.ainvoke({"messages": [{"role": "user", "content": task}]})
        return extract_agent_run_result(
            result,
            "The document worker could not produce a final response.",
        )
