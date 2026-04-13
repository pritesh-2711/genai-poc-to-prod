"""Single RAG agent backed by high-level tools.

The agent is intentionally narrow:
- memory resolution remains deterministic in the orchestrator
- retrieval/reranking stay hidden behind document tools
- the agent focuses on planning, tool selection, and synthesis
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from ..chat_service import ChatService
from ..core.logging import LoggingManager
from ..core.models import ChatRecord
from ._shared import extract_agent_run_result

logger = LoggingManager.get_logger(__name__)

_SINGLE_RAG_AGENT_PROMPT = """You are a research assistant agent.

You can use tools to:
- inspect uploaded session documents
- search inside uploaded documents
- summarize or extract metadata from uploaded papers
- search the live web for current information
- fetch webpages for deeper reading
- perform calculations when needed

Working style:
- Prefer uploaded documents when the question is answerable from them.
- Use web tools only when the answer needs external or current information.
- Use multiple tool calls when needed, but avoid redundant calls.
- Synthesize tool outputs into a clear final answer.
- If evidence is incomplete, say what you found and what is uncertain.
- Do not expose raw chain-of-thought.
"""

from ._shared import AgentRunResult as SingleRAGAgentResult


class SingleRAGAgent:
    """Small wrapper around LangChain's create_agent API."""

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
        return f"{base}\n\n{_SINGLE_RAG_AGENT_PROMPT}"

    async def arun(self, user_message: str) -> SingleRAGAgentResult:
        model = self._chat_service.llm_provider.llm
        graph = create_agent(
            model=model,
            tools=self._tools,
            system_prompt=self._build_system_prompt(),
            name="single_rag_agent",
        )

        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]}
        )
        extracted = extract_agent_run_result(
            result,
            "I couldn't produce a final response from the agent run.",
        )
        if extracted.response.startswith("I couldn't"):
            logger.warning("SingleRAGAgent produced no assistant content.")
        return SingleRAGAgentResult(
            response=extracted.response,
            tools_used=extracted.tools_used,
            step_count=extracted.step_count,
        )
