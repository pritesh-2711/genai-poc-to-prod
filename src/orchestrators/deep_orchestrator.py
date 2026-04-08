"""Deep-mode subgraph orchestrator.

Graph topology:
    START
      → resolve_memory
      → analyze_query              (LLM: intent + clear_intent + complexity)
      → [clear_intent router]
          → False: query_clarification  → interrupt() → resume with user input → retrieve
          → True:  [complexity router]
              → low:      retrieve
              → moderate: query_rewrite → retrieve
              → high:     query_decompose → [Send × N] → retrieve_sub_query (parallel)
      → rerank_and_build_context
      → generate
      → [validation router]
          → pass (or iter >= MAX_ITER): finalize
          → fail:                        correction → generate  (loop, max 3 times)
      → END

Key design decisions:
  - "low" complexity skips analysis overhead and goes straight to retrieve,
    behaving identically to fast mode from that point forward.
  - "high" complexity uses LangGraph's Send API to fan out N sub-query
    retrievals in parallel, then accumulates via the operator.add reducer.
  - Self-validation only runs in deep mode; fast mode has no correction loop.
  - interrupt() pauses the graph when the query is ambiguous; the API resumes
    it with the user's clarification in the next request.
"""

import json
import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send, interrupt

from .base import BaseOrchestrator
from .state import RAGState, SubQueryState

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 3

# ── LLM prompt templates ──────────────────────────────────────────────────────

_ANALYZE_QUERY_PROMPT = """\
You are a query analysis expert for a RAG (retrieval-augmented generation) system.

Analyze the following user query and return a JSON object with these fields:
- query_intent:          What the user wants to know (1–2 sentences).
- clear_intent:          true if the query is clear enough to search for, false if ambiguous/incomplete.
- query_complexity:      "low" | "moderate" | "high"
    low:      Simple factual question about a single concept.
    moderate: Requires rephrasing for better retrieval (vague wording, implicit context).
    high:     Complex multi-part question that benefits from decomposition into sub-queries.
- query_status:          "clear" | "ambiguous" | "incomplete"
- clarification_question: (only when clear_intent is false) What to ask the user to clarify.

User query: {query}

Respond ONLY with the JSON object — no markdown fences, no extra text.
"""

_QUERY_REWRITE_PROMPT = """\
You are a query optimization expert for document retrieval.
Rewrite the query below to maximise retrieval recall. Focus on: key concepts, specific \
terminology, and searchable noun phrases.

Original query: {query}
Query intent:   {intent}

Respond with ONLY the rewritten query — no explanation.
"""

_QUERY_DECOMPOSE_PROMPT = """\
Break the complex query below into 2–4 focused sub-queries that together cover the full topic.
Each sub-query should be independently searchable in a document store.

Original query: {query}
Query intent:   {intent}

Respond ONLY with a JSON object: {{"sub_queries": ["sub-query 1", "sub-query 2", ...]}}
"""

_VALIDATE_RESPONSE_PROMPT = """\
You are a strict quality evaluator for a RAG system.

Evaluate whether the assistant's answer adequately addresses the user's question using \
the provided context.

User question: {question}

Retrieved context:
{context}

Assistant answer:
{answer}

Evaluation criteria:
1. The answer directly addresses the question.
2. The answer is grounded in the provided context (no hallucination).
3. The answer is complete — it does not omit key information present in the context.

Respond ONLY with a JSON object: {{"result": "pass" | "fail", "note": "brief reason if fail, else empty string"}}
"""


class DeepOrchestrator(BaseOrchestrator):
    """Compiles the deep-mode RAG subgraph.

    Inherits shared nodes from BaseOrchestrator and adds deep-mode-specific
    nodes: analyze_query, query_clarification, query_rewrite, query_decompose,
    validate_response, and correction.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Deep-mode-only nodes
    # ──────────────────────────────────────────────────────────────────────────

    async def _analyze_query_node(self, state: RAGState) -> dict:
        """LLM call: classify intent, detect clarity, determine complexity."""
        query = state["original_query"]
        prompt = _ANALYZE_QUERY_PROMPT.format(query=query)

        raw = await self._chat_service.get_response_async(user_message=prompt)

        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            logger.warning(f"[analyze_query] JSON parse failed, defaulting to clear/low. raw={raw!r}")
            data = {
                "query_intent": query,
                "clear_intent": True,
                "query_complexity": "low",
                "query_status": "clear",
                "clarification_question": "",
            }

        return {
            "query_intent": data.get("query_intent", query),
            "clear_intent": bool(data.get("clear_intent", True)),
            "query_complexity": data.get("query_complexity", "low"),
            "query_status": data.get("query_status", "clear"),
            "clarification_question": data.get("clarification_question", ""),
            # Set retrieval_query to original by default; may be overwritten by rewrite/decompose
            "retrieval_query": query,
        }

    async def _query_clarification_node(self, state: RAGState) -> dict:
        """Pause graph with interrupt(); resume when user provides clarification.

        LangGraph's interrupt() saves graph state to the checkpointer and
        raises an interrupt.  The caller (RAGOrchestrator) detects this and
        returns the clarification question as the assistant reply.  On the
        next request the API calls graph.ainvoke(Command(resume=...)) to
        continue from this exact point.

        After resuming, the clarified query is treated as the retrieval query
        with low complexity (direct retrieval — no further analysis needed).
        """
        clarification_question = (
            state.get("clarification_question")
            or "Could you please clarify your question? More detail will help me give a better answer."
        )

        # Pause here — returns user's reply when resumed
        user_clarification: str = interrupt(clarification_question)

        logger.info(f"[clarification] resumed with: {user_clarification!r}")
        return {
            "original_query": user_clarification,
            "retrieval_query": user_clarification,
            "clear_intent": True,
            "query_complexity": "low",
        }

    async def _query_rewrite_node(self, state: RAGState) -> dict:
        """LLM call: rewrite the query for better retrieval (moderate complexity)."""
        prompt = _QUERY_REWRITE_PROMPT.format(
            query=state["original_query"],
            intent=state.get("query_intent", ""),
        )
        reframed = await self._chat_service.get_response_async(user_message=prompt)
        reframed = reframed.strip()
        logger.debug(f"[query_rewrite] '{state['original_query']}' → '{reframed}'")
        return {"reframed_query": reframed, "retrieval_query": reframed}

    async def _query_decompose_node(self, state: RAGState) -> dict:
        """LLM call: decompose into sub-queries (high complexity, Send API fan-out)."""
        prompt = _QUERY_DECOMPOSE_PROMPT.format(
            query=state["original_query"],
            intent=state.get("query_intent", ""),
        )
        raw = await self._chat_service.get_response_async(user_message=prompt)

        try:
            data = json.loads(raw.strip())
            sub_queries: list[str] = data.get("sub_queries", [])
        except json.JSONDecodeError:
            logger.warning(f"[query_decompose] JSON parse failed, using original. raw={raw!r}")
            sub_queries = [state["original_query"]]

        if not sub_queries:
            sub_queries = [state["original_query"]]

        logger.info(f"[query_decompose] {len(sub_queries)} sub-queries: {sub_queries}")
        return {"sub_queries": sub_queries}

    async def _validate_response_node(self, state: RAGState) -> dict:
        """LLM-as-judge: evaluate whether the generated answer passes quality criteria."""
        prompt = _VALIDATE_RESPONSE_PROMPT.format(
            question=state.get("original_query", ""),
            context=state.get("rag_context", "(no context)"),
            answer=state.get("llm_response", ""),
        )
        raw = await self._chat_service.get_response_async(user_message=prompt)

        try:
            data = json.loads(raw.strip())
            result: Literal["pass", "fail"] = data.get("result", "pass")
            note: str = data.get("note", "")
        except json.JSONDecodeError:
            logger.warning(f"[validate] JSON parse failed, defaulting pass. raw={raw!r}")
            result = "pass"
            note = ""

        iteration = state.get("iteration_count", 0)
        logger.info(f"[validate] iter={iteration} result={result} note={note!r}")

        # Track best response — update whenever we have a new llm_response
        llm_response = state.get("llm_response", "")
        best_response = state.get("best_response", "") or llm_response

        return {
            "validation_result": result,
            "correction_note": note,
            "best_response": best_response,
            "iteration_count": iteration + 1,
        }

    async def _correction_node(self, state: RAGState) -> dict:
        """Prepare state for the next generation attempt after a failed validation.

        Clears llm_response so generate node knows it's a fresh attempt.
        The correction_note from validation is already in state and will be
        prepended to the user message in _generate_node.
        """
        logger.debug(
            f"[correction] iter={state.get('iteration_count', 0)} "
            f"note={state.get('correction_note', '')!r}"
        )
        return {"llm_response": ""}

    # ──────────────────────────────────────────────────────────────────────────
    # Conditional edge routing functions
    # ──────────────────────────────────────────────────────────────────────────

    def _route_intent(self, state: RAGState) -> Literal["query_clarification", "route_complexity"]:
        """After analyze_query: route on whether the intent is clear."""
        if not state.get("clear_intent", True):
            return "query_clarification"
        return "route_complexity"

    def _route_complexity(self, state: RAGState) -> Literal["retrieve", "query_rewrite", "query_decompose"]:
        """After clarity confirmed: route by query complexity."""
        complexity = state.get("query_complexity", "low")
        if complexity == "high":
            return "query_decompose"
        if complexity == "moderate":
            return "query_rewrite"
        # low complexity → same as fast mode from here
        return "retrieve"

    def _fan_out_sub_queries(self, state: RAGState) -> list[Send]:
        """Send API: fan out one retrieve_sub_query task per sub-query (parallel)."""
        return [
            Send(
                "retrieve_sub_query",
                SubQueryState(
                    sub_query=sq,
                    session_id=state["session_id"],
                    query_embedding=state.get("query_embedding", []),
                ),
            )
            for sq in state.get("sub_queries", [])
        ]

    def _route_validation(
        self, state: RAGState
    ) -> Literal["correction", "finalize"]:
        """After validate_response: decide whether to retry or finalize."""
        iteration = state.get("iteration_count", 0)
        result = state.get("validation_result", "pass")

        if result == "pass" or iteration >= _MAX_ITERATIONS:
            return "finalize"
        return "correction"

    async def _finalize_node(self, state: RAGState) -> dict:
        """Set final_response from the best available answer."""
        # Use llm_response if validation passed; otherwise fall back to best_response
        if state.get("validation_result") == "pass":
            final = state.get("llm_response", "")
        else:
            final = state.get("best_response") or state.get("llm_response", "")
        return {"final_response": final}

    # ──────────────────────────────────────────────────────────────────────────
    # Graph compilation
    # ──────────────────────────────────────────────────────────────────────────

    def build_graph(self) -> CompiledStateGraph:
        """Build and compile the deep-mode subgraph.

        Returns:
            A compiled LangGraph subgraph.  The checkpointer is NOT attached
            here — it is attached at the RAGOrchestrator level so that
            interrupt() state persists across HTTP requests.
        """
        builder = StateGraph(RAGState)

        # ── Register all nodes ────────────────────────────────────────────────
        builder.add_node("resolve_memory", self._resolve_memory_node)
        builder.add_node("analyze_query", self._analyze_query_node)
        builder.add_node("query_clarification", self._query_clarification_node)
        builder.add_node("route_complexity", lambda s: s)        # pass-through routing node
        builder.add_node("query_rewrite", self._query_rewrite_node)
        builder.add_node("query_decompose", self._query_decompose_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("retrieve_sub_query", self._retrieve_sub_query_node)
        builder.add_node("rerank_and_build_context", self._rerank_and_build_context_node)
        builder.add_node("generate", self._generate_node)
        builder.add_node("validate_response", self._validate_response_node)
        builder.add_node("correction", self._correction_node)
        builder.add_node("finalize", self._finalize_node)

        # ── Linear opening ────────────────────────────────────────────────────
        builder.add_edge(START, "resolve_memory")
        builder.add_edge("resolve_memory", "analyze_query")

        # ── Intent routing ────────────────────────────────────────────────────
        builder.add_conditional_edges(
            "analyze_query",
            self._route_intent,
            {
                "query_clarification": "query_clarification",
                "route_complexity": "route_complexity",
            },
        )

        # After clarification (interrupt/resume) → go directly to retrieve
        builder.add_edge("query_clarification", "retrieve")

        # ── Complexity routing ────────────────────────────────────────────────
        builder.add_conditional_edges(
            "route_complexity",
            self._route_complexity,
            {
                "retrieve": "retrieve",
                "query_rewrite": "query_rewrite",
                "query_decompose": "query_decompose",
            },
        )

        builder.add_edge("query_rewrite", "retrieve")

        # High complexity: fan-out via Send API, accumulate via operator.add
        builder.add_conditional_edges("query_decompose", self._fan_out_sub_queries)
        builder.add_edge("retrieve_sub_query", "rerank_and_build_context")

        # Single-query paths also merge into rerank
        builder.add_edge("retrieve", "rerank_and_build_context")

        # ── Generation + validation loop ─────────────────────────────────────
        builder.add_edge("rerank_and_build_context", "generate")
        builder.add_edge("generate", "validate_response")

        builder.add_conditional_edges(
            "validate_response",
            self._route_validation,
            {
                "correction": "correction",
                "finalize": "finalize",
            },
        )

        # Correction loops back to generate (not re-retrieval — context is unchanged)
        builder.add_edge("correction", "generate")

        builder.add_edge("finalize", END)

        graph = builder.compile()
        logger.info("DeepOrchestrator subgraph compiled.")
        return graph
