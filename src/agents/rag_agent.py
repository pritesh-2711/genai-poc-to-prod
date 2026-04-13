"""Single RAG agent backed by high-level tools.

The agent is intentionally narrow:
- memory resolution remains deterministic in the orchestrator
- retrieval/reranking stay hidden behind document tools
- the agent focuses on planning, tool selection, and synthesis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent

from ..chat_service import ChatService
from ..core.logging import LoggingManager
from ..core.models import ChatRecord

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


@dataclass
class SingleRAGAgentResult:
    """Final output captured from one agent run."""

    response: str
    tools_used: list[str]
    step_count: int


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

        messages = result.get("messages", []) if isinstance(result, dict) else []
        tools_used: list[str] = []
        final_response = ""

        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None) or []
            for call in tool_calls:
                name = call.get("name")
                if name:
                    tools_used.append(name)

        for msg in reversed(messages):
            msg_type = getattr(msg, "type", "")
            content = getattr(msg, "content", "")
            if msg_type == "ai" and isinstance(content, str) and content.strip():
                final_response = content.strip()
                break

        if not final_response:
            logger.warning("SingleRAGAgent produced no assistant content.")
            final_response = "I couldn't produce a final response from the agent run."

        return SingleRAGAgentResult(
            response=final_response,
            tools_used=tools_used,
            step_count=max(len(tools_used), 1),
        )
