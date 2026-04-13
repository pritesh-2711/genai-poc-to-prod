"""
Stateless utility tools that require no session context or external APIs.

These can be imported and used directly without going through build_document_tools().
"""

import math
import re

from langchain_core.tools import tool

from ..core.logging import LoggingManager

logger = LoggingManager.get_logger(__name__)

_SAFE_MATH_GLOBALS: dict = {"__builtins__": {}}
_SAFE_MATH_LOCALS: dict = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_")
}


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Use this when a research paper contains numerical claims, statistics, or
    formulas that need to be verified or computed. Also useful for unit
    conversions, percentage calculations, and comparing reported metrics.

    Supports standard arithmetic, exponentiation, and Python's math module
    functions (sqrt, log, exp, sin, cos, ceil, floor, etc.).

    Examples:
        "2 ** 32"                -> "4294967296"
        "sqrt(2)"                -> "1.4142135623730951"
        "log(1000, 10)"          -> "2.9999999999999996"
        "(0.92 - 0.87) / 0.87"  -> "0.057471264367816... (5.75% improvement)"

    Args:
        expression: A mathematical expression as a string. Do not include
                    assignment operators (=) or print statements.

    Returns:
        The computed result as a string, or an error message if the expression
        is invalid or unsafe.
    """
    expression = expression.strip()
    logger.debug("calculate: expression=%r", expression)

    if re.search(r"[;]|import|exec|eval|open|os\.|sys\.", expression):
        return "Expression rejected: contains disallowed keywords or operators."

    try:
        result = eval(expression, _SAFE_MATH_GLOBALS, _SAFE_MATH_LOCALS)  # noqa: S307
    except ZeroDivisionError:
        return "Error: division by zero."
    except (SyntaxError, NameError, TypeError, ValueError) as exc:
        return f"Could not evaluate expression: {exc}"
    except Exception as exc:
        return f"Unexpected error: {exc}"

    if isinstance(result, float):
        formatted = f"{result:.10g}"
    else:
        formatted = str(result)

    logger.debug("calculate: result=%s", formatted)
    return formatted