"""Shared helpers for agent wrappers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentRunResult:
    """Normalized result extracted from a LangChain agent run."""

    response: str
    tools_used: list[str]
    step_count: int
    charts: list[str] = field(default_factory=list)


def _extract_charts_from_messages(messages: list[Any]) -> list[str]:
    """Scan ToolMessages for base64 PNG charts returned by the analyse tool.

    Handles both plain-string content (most adapters) and list-of-content-block
    content (some MCP adapter versions).
    """
    charts: list[str] = []
    for msg in messages:
        if getattr(msg, "type", "") != "tool":
            continue
        raw_content = getattr(msg, "content", "")

        # Normalise to a list of strings to try
        candidates: list[str] = []
        if isinstance(raw_content, str):
            candidates = [raw_content]
        elif isinstance(raw_content, list):
            for item in raw_content:
                if isinstance(item, str):
                    candidates.append(item)
                elif isinstance(item, dict):
                    # MCP TextContent block: {"type": "text", "text": "..."}
                    text = item.get("text") or item.get("content", "")
                    if isinstance(text, str):
                        candidates.append(text)

        for text in candidates:
            if '"charts"' not in text:
                continue
            try:
                data = json.loads(text)
                if isinstance(data, dict) and isinstance(data.get("charts"), list):
                    charts.extend(c for c in data["charts"] if isinstance(c, str) and c)
            except Exception as exc:
                logger.debug("_extract_charts_from_messages: parse failed: %s", exc)
    return charts


def extract_agent_run_result(
    result: dict[str, Any] | Any,
    empty_response_message: str,
) -> AgentRunResult:
    """Extract final AI content, tool names, and charts from an agent invocation result."""
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

    charts = _extract_charts_from_messages(messages)

    return AgentRunResult(
        response=final_response,
        tools_used=tools_used,
        step_count=max(len(tools_used), 1),
        charts=charts,
    )
