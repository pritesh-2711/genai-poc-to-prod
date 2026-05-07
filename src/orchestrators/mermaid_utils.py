"""Mermaid diagram syntax validation and LLM self-correction.

Exported helpers:
    is_valid_mermaid(code)           — structural check (no network calls)
    fix_mermaid_in_text(text, svc)   — async; corrects invalid blocks via one LLM retry
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..chat_service import ChatService

logger = logging.getLogger(__name__)

# Matches ```mermaid ... ``` blocks (lazy, DOTALL)
_BLOCK_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)

# Every valid Mermaid diagram starts with one of these keywords
_VALID_STARTS = {
    "graph",
    "flowchart",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "stateDiagram-v2",
    "erDiagram",
    "gantt",
    "pie",
    "gitGraph",
    "journey",
    "mindmap",
    "timeline",
    "xychart-beta",
    "block-beta",
    "quadrantChart",
}

# Patterns that reliably break Mermaid.js even when the keyword is valid
_BAD_PATTERNS = re.compile(
    r"&(?:amp|lt|gt|quot|apos|nbsp);|"   # HTML entities
    r"<(?!br\s*/?>)[a-zA-Z]|"             # HTML tags (not <br>)
    r"\bsubgraph\b(?!.*\bend\b)",          # unclosed subgraph (no matching 'end')
    re.DOTALL,
)

_FALLBACK = ""


def is_valid_mermaid(code: str) -> bool:
    """Return True if the Mermaid code has a valid first keyword AND no known
    body-level patterns that break Mermaid.js rendering.

    Skips blank lines and %% comments before checking the first real line.
    """
    first_word_ok = False
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            continue
        first_word = re.split(r"[\s:(-]", stripped)[0]
        first_word_ok = first_word in _VALID_STARTS
        break

    if not first_word_ok:
        return False

    # Count subgraph / end pairs — mismatches break rendering
    subgraph_count = len(re.findall(r"^\s*subgraph\b", code, re.MULTILINE))
    end_count = len(re.findall(r"^\s*end\b", code, re.MULTILINE))
    if subgraph_count != end_count:
        return False

    # Reject known bad patterns (HTML entities, unclosed tags, etc.)
    if _BAD_PATTERNS.search(code):
        return False

    return True


async def fix_mermaid_in_text(text: str, chat_service: "ChatService") -> str:
    """Find all ```mermaid``` blocks in *text*, validate each, and attempt one
    LLM correction pass for any that are structurally invalid.

    Invalid blocks that cannot be corrected are replaced with a plain-text
    fallback so they don't crash the frontend renderer.
    """
    if "```mermaid" not in text:
        return text

    matches = list(_BLOCK_RE.finditer(text))
    if not matches:
        return text

    result = text
    offset = 0  # tracks how replacements shift string positions

    for m in matches:
        code = m.group(1)
        if is_valid_mermaid(code):
            continue  # nothing to fix

        replacement = _FALLBACK
        correction_prompt = (
            "The following Mermaid diagram has a syntax error. "
            "Fix it and return ONLY the corrected ```mermaid ... ``` code block — nothing else.\n\n"
            "Common issues to fix:\n"
            "- Replace HTML entities (&amp; &lt; &gt;) with plain text (and < >)\n"
            "- Remove HTML tags from node labels\n"
            "- Every subgraph block must have a matching `end` keyword\n"
            "- Node labels must use [] () {} or (()) shapes — no raw special chars\n"
            "- Keep node IDs short and alphanumeric (no spaces in IDs)\n\n"
            f"```mermaid\n{code.strip()}\n```"
        )

        try:
            fixed_text = await chat_service.get_response_async(
                user_message=correction_prompt,
                short_term_history=[],
                long_term_history=[],
                rag_context=None,
            )
            fixed_match = _BLOCK_RE.search(fixed_text)
            if fixed_match and is_valid_mermaid(fixed_match.group(1)):
                replacement = fixed_match.group(0)
                logger.info("mermaid_utils: corrected invalid block via LLM retry")
            else:
                logger.warning("mermaid_utils: LLM retry did not produce valid mermaid, using fallback")
        except Exception as exc:
            logger.warning("mermaid_utils: LLM correction failed (%s), using fallback", exc)

        start = m.start() + offset
        end = m.end() + offset
        result = result[:start] + replacement + result[end:]
        offset += len(replacement) - (m.end() - m.start())

    return result
