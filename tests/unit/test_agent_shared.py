"""Unit tests for shared agent helpers.

Tests cover:
- _extract_charts_from_messages  (plain string, list-of-TextContent, no-charts)
- extract_agent_run_result        (final response, tools_used, fallback)
"""

import json
from unittest.mock import MagicMock

from src.agents._shared import (
    _extract_charts_from_messages,
    extract_agent_run_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_msg(content):
    msg = MagicMock()
    msg.type = "tool"
    msg.content = content
    msg.tool_calls = []
    return msg


def _ai_msg(content, tool_calls=None):
    msg = MagicMock()
    msg.type = "ai"
    msg.content = content
    msg.tool_calls = tool_calls or []
    return msg


def _human_msg(content):
    msg = MagicMock()
    msg.type = "human"
    msg.content = content
    msg.tool_calls = []
    return msg


# ---------------------------------------------------------------------------
# _extract_charts_from_messages
# ---------------------------------------------------------------------------

def test_extracts_charts_from_string_content() -> None:
    payload = json.dumps({"charts": ["base64abc", "base64def"]})
    charts = _extract_charts_from_messages([_tool_msg(payload)])
    assert charts == ["base64abc", "base64def"]


def test_extracts_charts_from_list_of_text_content() -> None:
    payload = json.dumps({"charts": ["base64xyz"]})
    content = [{"type": "text", "text": payload}]
    charts = _extract_charts_from_messages([_tool_msg(content)])
    assert charts == ["base64xyz"]


def test_extracts_charts_from_list_with_content_key() -> None:
    payload = json.dumps({"charts": ["base64pqr"]})
    content = [{"content": payload}]
    charts = _extract_charts_from_messages([_tool_msg(content)])
    assert charts == ["base64pqr"]


def test_returns_empty_when_no_charts_key() -> None:
    payload = json.dumps({"result": "no charts here"})
    charts = _extract_charts_from_messages([_tool_msg(payload)])
    assert charts == []


def test_returns_empty_for_empty_charts_list() -> None:
    payload = json.dumps({"charts": []})
    charts = _extract_charts_from_messages([_tool_msg(payload)])
    assert charts == []


def test_ignores_non_tool_messages() -> None:
    payload = json.dumps({"charts": ["should_not_appear"]})
    charts = _extract_charts_from_messages([_ai_msg(payload), _human_msg(payload)])
    assert charts == []


def test_ignores_empty_chart_strings() -> None:
    payload = json.dumps({"charts": ["valid", "", "  "]})
    charts = _extract_charts_from_messages([_tool_msg(payload)])
    # Only non-empty strings should be included
    assert "valid" in charts
    assert "" not in charts


def test_handles_invalid_json_gracefully() -> None:
    # Should not raise; silently skip the message
    charts = _extract_charts_from_messages([_tool_msg('{"charts": [broken json')])
    assert charts == []


# ---------------------------------------------------------------------------
# extract_agent_run_result
# ---------------------------------------------------------------------------

def test_returns_last_ai_message_as_final_response() -> None:
    msgs = [_ai_msg("first answer"), _ai_msg("final answer")]
    result = extract_agent_run_result({"messages": msgs}, "no response")
    assert result.response == "final answer"


def test_skips_whitespace_only_ai_messages() -> None:
    msgs = [_ai_msg("real answer"), _ai_msg("   ")]
    result = extract_agent_run_result({"messages": msgs}, "no response")
    assert result.response == "real answer"


def test_collects_all_tool_call_names() -> None:
    tool_msg_ai = _ai_msg(
        "thinking",
        tool_calls=[{"name": "web_search"}, {"name": "calculate"}],
    )
    final = _ai_msg("done")
    result = extract_agent_run_result({"messages": [tool_msg_ai, final]}, "no response")
    assert "web_search" in result.tools_used
    assert "calculate" in result.tools_used


def test_tools_used_order_matches_call_order() -> None:
    step1 = _ai_msg("step1", tool_calls=[{"name": "tool_a"}])
    step2 = _ai_msg("step2", tool_calls=[{"name": "tool_b"}])
    final = _ai_msg("done")
    result = extract_agent_run_result({"messages": [step1, step2, final]}, "no response")
    assert result.tools_used == ["tool_a", "tool_b"]


def test_fallback_when_no_ai_message_found() -> None:
    msgs = [_tool_msg("some tool output")]
    result = extract_agent_run_result({"messages": msgs}, "I could not find an answer.")
    assert result.response == "I could not find an answer."


def test_charts_are_collected_from_tool_messages() -> None:
    payload = json.dumps({"charts": ["b64chart"]})
    msgs = [_tool_msg(payload), _ai_msg("here is the analysis")]
    result = extract_agent_run_result({"messages": msgs}, "no response")
    assert "b64chart" in result.charts


def test_step_count_is_at_least_one() -> None:
    result = extract_agent_run_result({"messages": [_ai_msg("hello")]}, "no response")
    assert result.step_count >= 1


def test_empty_messages_returns_fallback() -> None:
    result = extract_agent_run_result({"messages": []}, "fallback message")
    assert result.response == "fallback message"


def test_non_dict_result_returns_fallback() -> None:
    result = extract_agent_run_result("not a dict", "fallback message")
    assert result.response == "fallback message"
