"""Worker agent specialized in numerical reasoning and exact calculations."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from ...chat_service import ChatService
from ...core.models import ChatRecord
from .._shared import AgentRunResult, extract_agent_run_result

_COMPUTATION_PROMPT = """You are the Computation Worker.

Role:
- Solve only the numerical or formula-based part of the task.
- Use the calculation tool for exact answers whenever possible.
- Return the calculation result clearly and concisely.
- Do not speculate beyond what the math supports.
"""


class ComputationWorkerAgent:
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
        return f"{base}\n\n{_COMPUTATION_PROMPT}"

    async def arun(self, task: str) -> AgentRunResult:
        graph = create_agent(
            model=self._chat_service.llm_provider.llm,
            tools=self._tools,
            system_prompt=self._build_system_prompt(),
            name="computation_worker",
        )
        result = await graph.ainvoke({"messages": [{"role": "user", "content": task}]})
        return extract_agent_run_result(
            result,
            "The computation worker could not produce a final response.",
        )
