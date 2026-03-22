"""Guardrails package — safety checks that run before/after the LLM call."""

from .input_guard import GuardResult, InputGuard

__all__ = ["InputGuard", "GuardResult"]
