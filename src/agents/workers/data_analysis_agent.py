"""Worker agent specialized in data analysis using the E2B sandbox (analyse tool)."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from ...chat_service import ChatService
from ...core.models import ChatRecord
from .._shared import AgentRunResult, extract_agent_run_result

_DATA_ANALYSIS_PROMPT = """You are the Data Analysis Worker.

Role:
- Perform quantitative analysis, statistics, or data exploration using Python code
  executed in a secure E2B sandbox via the `analyse` tool.
- Work only with data that has been explicitly provided to you (CSV text, numbers,
  or structured content from prior tool calls).
- Return clear findings — numbers, trends, comparisons — for the supervisor to synthesize.
- If the data is insufficient or the question is unanswerable with what was provided,
  say so clearly.

Working style:
- Write concise, correct pandas/scipy/numpy code in the `python_code` parameter.
- Always pass the `question` parameter so the sandbox output is self-documenting.
- If CSV data is available, pass it as `dataset_csv`.
- Keep code focused: one analysis per tool call rather than sprawling scripts.
- Quote specific numbers from the sandbox output in your findings.
- Do not fabricate results — only report what the sandbox actually returned.
"""


class DataAnalysisWorkerAgent:
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
        return f"{base}\n\n{_DATA_ANALYSIS_PROMPT}"

    async def arun(self, task: str) -> AgentRunResult:
        graph = create_agent(
            model=self._chat_service.llm_provider.llm,
            tools=self._tools,
            system_prompt=self._build_system_prompt(),
            name="data_analysis_worker",
        )
        result = await graph.ainvoke({"messages": [{"role": "user", "content": task}]})
        return extract_agent_run_result(
            result,
            "The data analysis worker could not produce a final response.",
        )
