"""Shared state definitions for the RAG LangGraph orchestrators.

RAGState is the main graph state — it flows through every node in both the
fast and deep subgraphs.  SubQueryState is the per-task state sent to the
parallel retrieve node via LangGraph's Send API in deep/high-complexity mode.

Design notes:
- ``raw_chunks`` uses ``operator.add`` so parallel Send tasks can safely
  accumulate retrieval results into a single list (map phase of map-reduce).
- All fields are Optional-equivalent (using default None or empty values) so
  the initial state dict passed from the API only needs to supply the inputs.
"""

import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict


class RAGState(TypedDict, total=False):
    """Full graph state shared across fast and deep orchestrators.

    Fields are grouped by concern.  The API sets the ``# Input`` fields
    before invoking the graph; all other fields are populated by nodes.
    """

    # ── Input (set by API before graph invocation) ──────────────────────────
    original_query: str
    query_embedding: list[float]        # pre-computed by API; reused in retrieve + memory
    category: Literal["workflow", "agent"]
    variant: Literal["fast", "deep", "single_rag_agent"]
    session_id: str
    user_id: str
    user_chat_id: str                   # UUID of the persisted user message (for memory dedup)

    # ── Deep mode — query analysis (analyze_query LLM node) ─────────────────
    query_intent: str
    clear_intent: bool                  # False → triggers query_clarification + interrupt
    query_status: str                   # "clear" | "ambiguous" | "incomplete"
    clarification_question: str         # populated when clear_intent=False

    # ── Deep mode — complexity routing ──────────────────────────────────────
    query_complexity: Literal["low", "moderate", "high"]

    # ── Deep mode — query transformation ────────────────────────────────────
    sub_queries: list[str]              # high complexity: decomposed sub-queries
    reframed_query: str                 # moderate complexity: rewritten query
    retrieval_query: str                # actual query string used for single retrieval
                                        # fast/low/moderate paths; high uses sub_queries

    # ── Retrieval ────────────────────────────────────────────────────────────
    # operator.add lets parallel Send tasks accumulate chunks without conflict
    raw_chunks: Annotated[list[dict], operator.add]

    # ── Post-retrieval ────────────────────────────────────────────────────────
    reranked_chunks: list[dict]
    rag_context: str                    # formatted string injected into system prompt

    # ── Memory (generation-only; NOT used in retrieval) ──────────────────────
    short_term_history: list
    long_term_history: list

    # ── Generation ────────────────────────────────────────────────────────────
    llm_response: str
    best_response: str                  # best answer seen so far across correction iterations

    # ── Deep mode — self-validation loop ─────────────────────────────────────
    validation_result: Literal["pass", "fail", ""]
    correction_note: str                # guidance fed back into the next generate call
    iteration_count: int

    # ── Output ───────────────────────────────────────────────────────────────
    final_response: str
    tools_used: list[str]
    agent_step_count: int


class SubQueryState(TypedDict):
    """Per-task state sent to the parallel retrieve node via Send API.

    The retrieve node returns ``{"raw_chunks": [...]}`` which LangGraph
    merges back into ``RAGState.raw_chunks`` via the ``operator.add`` reducer.
    """

    sub_query: str
    session_id: str
    query_embedding: list[float]        # ignored for sub-queries (node embeds sub_query itself)
