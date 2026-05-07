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

Chart / visualisation rules (CRITICAL — follow exactly):
- When generating a chart or plot, always call `plt.show()` at the end of your code.
  NEVER use `plt.savefig()` — it saves to a file the sandbox cannot return.
- `plt.show()` is the ONLY way charts are captured and displayed to the user.
- Always import matplotlib as: `import matplotlib.pyplot as plt`
- Always set figure size to (8, 5): `plt.figure(figsize=(8, 5))`
- Set a title and axis labels on every chart so it is self-explanatory.
- Example pattern:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title('Error Rates by System')
    plt.xlabel('System')
    plt.ylabel('Error Rate (%)')
    plt.tight_layout()
    plt.show()

Response rules when a chart is generated:
- The chart image is rendered directly in the UI — do NOT repeat the underlying data
  as a text list. The user can see it in the chart.
- After generating a chart, respond with ONE short sentence confirming what was shown,
  followed by at most 2 key insights or observations from the data.
- Do NOT list every data point in text. Do NOT offer to share the Python code
  unless the user explicitly asks for it.
- Example good response: "The bar chart shows error rates across all systems.
  LlamaParse and RaV-IDP gate_only stand out with the highest error rates at 30%
  and 29.7% respectively, while most other systems are near 0%."
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
