"""Shared helpers for agent wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentRunResult:
    """Normalized result extracted from a LangChain agent run."""

    response: str
    tools_used: list[str]
    step_count: int


def extract_agent_run_result(
    result: dict[str, Any] | Any,
    empty_response_message: str,
) -> AgentRunResult:
    """Extract final AI content and tool names from an agent invocation result."""
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
        final_response = empty_response_message

    return AgentRunResult(
        response=final_response,
        tools_used=tools_used,
        step_count=max(len(tools_used), 1),
    )
