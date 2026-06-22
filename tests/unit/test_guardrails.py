"""Unit tests for InputGuard.

DeepEval metrics (ToxicityMetric, BiasMetric, GEval) are fully mocked so
these tests are fast, free, and deterministic.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch

from src.core.models import GuardrailsConfig
from src.guardrails.input_guard import InputGuard, GuardResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cfg(toxicity=True, bias=False, injection=False, jailbreak=False) -> GuardrailsConfig:
    return GuardrailsConfig(
        enabled=True,
        toxicity=toxicity,
        bias=bias,
        prompt_injection=injection,
        jailbreaking=jailbreak,
        evaluator_model="gpt-4o-mini",
    )


def _no_metrics_cfg() -> GuardrailsConfig:
    return GuardrailsConfig(
        enabled=False,
        toxicity=False,
        bias=False,
        prompt_injection=False,
        jailbreaking=False,
        evaluator_model="gpt-4o-mini",
    )


LONG_CLEAN = "Tell me about the history of natural language processing research in academia."
LONG_BAD = "This is a very long message designed to trigger the toxicity guardrail in testing."


# ---------------------------------------------------------------------------
# Guard instantiation
# ---------------------------------------------------------------------------

def test_no_metrics_configured_when_all_disabled() -> None:
    guard = InputGuard(_no_metrics_cfg())
    assert guard._metrics == []


def test_toxicity_metric_added_when_enabled() -> None:
    with patch("src.guardrails.input_guard.ToxicityMetric") as MockT:
        MockT.return_value = MagicMock()
        guard = InputGuard(_cfg(toxicity=True))
    assert len(guard._metrics) == 1


def test_geval_added_when_injection_enabled() -> None:
    with (
        patch("src.guardrails.input_guard.ToxicityMetric") as MockT,
        patch("src.guardrails.input_guard.GEval") as MockG,
    ):
        MockT.return_value = MagicMock()
        MockG.return_value = MagicMock()
        guard = InputGuard(_cfg(toxicity=True, injection=True))
    assert len(guard._metrics) == 2


# ---------------------------------------------------------------------------
# Sync check()
# ---------------------------------------------------------------------------

def test_check_passes_when_no_metrics() -> None:
    guard = InputGuard(_no_metrics_cfg())
    result = guard.check(LONG_CLEAN)
    assert result.passed is True


def test_check_short_message_bypasses_metrics() -> None:
    with patch("src.guardrails.input_guard.ToxicityMetric") as MockT:
        mock_metric = MagicMock()
        MockT.return_value = mock_metric
        guard = InputGuard(_cfg(toxicity=True))
        result = guard.check("hi")  # < 40 chars
    assert result.passed is True
    mock_metric.measure.assert_not_called()


def test_check_calls_metric_measure_for_long_input() -> None:
    with patch("src.guardrails.input_guard.ToxicityMetric") as MockT:
        mock_metric = MagicMock()
        mock_metric.is_successful.return_value = True
        MockT.return_value = mock_metric
        guard = InputGuard(_cfg(toxicity=True))
        guard.check(LONG_CLEAN)
    mock_metric.measure.assert_called_once()


def test_check_returns_passed_when_metric_succeeds() -> None:
    with patch("src.guardrails.input_guard.ToxicityMetric") as MockT:
        mock_metric = MagicMock()
        mock_metric.is_successful.return_value = True
        MockT.return_value = mock_metric
        guard = InputGuard(_cfg(toxicity=True))
        result = guard.check(LONG_CLEAN)
    assert result.passed is True
    assert result.violated_guard is None


def test_check_returns_blocked_when_metric_fails() -> None:
    with patch("src.guardrails.input_guard.ToxicityMetric") as MockT:
        mock_metric = MagicMock()
        mock_metric.is_successful.return_value = False
        mock_metric.__class__.__name__ = "ToxicityMetric"
        MockT.return_value = mock_metric
        guard = InputGuard(_cfg(toxicity=True))
        result = guard.check(LONG_BAD)
    assert result.passed is False
    assert result.violated_guard is not None


def test_check_stops_at_first_failing_metric() -> None:
    """Guard should short-circuit and not call subsequent metrics."""
    with (
        patch("src.guardrails.input_guard.ToxicityMetric") as MockT,
        patch("src.guardrails.input_guard.BiasMetric") as MockB,
    ):
        mock_toxicity = MagicMock()
        mock_toxicity.is_successful.return_value = False
        mock_bias = MagicMock()
        MockT.return_value = mock_toxicity
        MockB.return_value = mock_bias

        guard = InputGuard(_cfg(toxicity=True, bias=True))
        guard.check(LONG_BAD)

    # bias metric should never be evaluated after toxicity fails
    mock_bias.measure.assert_not_called()


# ---------------------------------------------------------------------------
# Async acheck()
# ---------------------------------------------------------------------------

def test_acheck_passes_when_no_metrics() -> None:
    guard = InputGuard(_no_metrics_cfg())
    result = asyncio.run(guard.acheck(LONG_CLEAN))
    assert result.passed is True


def test_acheck_short_message_bypasses_metrics() -> None:
    with patch("src.guardrails.input_guard.ToxicityMetric") as MockT:
        mock_metric = MagicMock()
        mock_metric.a_measure = MagicMock(return_value=None)
        MockT.return_value = mock_metric
        guard = InputGuard(_cfg(toxicity=True))
        result = asyncio.run(guard.acheck("hi"))
    assert result.passed is True
    mock_metric.a_measure.assert_not_called()


def test_acheck_returns_blocked_when_metric_fails() -> None:
    async def _fake_a_measure(_):
        pass

    with patch("src.guardrails.input_guard.ToxicityMetric") as MockT:
        mock_metric = MagicMock()
        mock_metric.a_measure.return_value = _fake_a_measure(None)
        mock_metric.is_successful.return_value = False
        mock_metric.__class__.__name__ = "ToxicityMetric"
        MockT.return_value = mock_metric
        guard = InputGuard(_cfg(toxicity=True))
        result = asyncio.run(guard.acheck(LONG_BAD))
    assert result.passed is False
