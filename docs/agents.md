# Agent Architecture

This project now supports two agentic chat variants alongside deterministic
workflow modes:

- `single_rag_agent`
- `supervisor_orchestration_agent`

Both are exposed through the same API contract:

```json
{
  "message": "How does this paper compare with recent industry practice?",
  "category": "agent",
  "variant": "supervisor_orchestration_agent"
}
```

---

## Why agents were added

The earlier workflow architecture was strong for deterministic RAG:

- `fast` handled straightforward document-grounded requests with low latency
- `deep` handled ambiguity, decomposition, and validation with fixed graph logic

But there are questions where fixed graph routing is not the best fit:

- The user may need a mix of uploaded-document evidence, current web evidence,
  and exact calculation in one answer.
- The best sequence of actions may depend on what the tools return.
- A fixed "retrieve once, then answer" pattern can be too rigid for research-style
  synthesis.

The agent layer addresses this by giving the model high-level tools and letting
it decide how to gather evidence before answering.

---

## Design principles

Three design decisions drive the implementation.

### 1. Keep infrastructure deterministic

Agents do **not** get direct access to low-level backend plumbing such as:

- embedding generation
- raw pgvector search
- reranker invocation
- parent-chunk fetching
- memory loading

Those remain fixed backend concerns.

This keeps the system easier to debug, cheaper to run, and less prone to
tool-ordering errors.

### 2. Expose only user-meaningful tools

The tools visible to the agent are high-level capabilities:

- `get_uploaded_documents`
- `search_documents`
- `summarize_document`
- `extract_paper_metadata`
- `web_search`
- `fetch_webpage`
- `calculate`

Each tool hides the lower-level retrieval/reranking mechanics it needs.

### 3. Preserve deterministic memory resolution

Even in agent mode, short-term and long-term memory are resolved by the
orchestrator before the agent runs. Memory is still injected into the system
prompt in a controlled way; the agent is responsible for planning and
tool selection, not memory loading.

---

## Single RAG Agent

### Purpose

`single_rag_agent` is one agent with access to all high-level tools.

It is best for:

- ordinary research chat
- document QA
- light web augmentation
- smaller multi-step questions

### Backend flow

```text
resolve_memory
  → run_agent
  → END
```

### Tool access

The single agent receives:

- all document tools
- all web tools
- the calculation tool

### Why this mode exists

It gives a flexible agentic path without the coordination overhead of a
supervisor-plus-workers setup.

This is the lowest-friction agent mode and the best first step before adding
true multi-agent delegation.

---

## Supervisor Orchestration Agent

### Purpose

`supervisor_orchestration_agent` is a multi-agent design:

- one supervisor
- three specialist workers

The supervisor plans the work, delegates to workers, and synthesizes the final
answer.

It is best for:

- document vs web comparisons
- multi-source research questions
- mixed evidence + calculation tasks
- questions where specialist separation improves reliability

### Backend flow

```text
resolve_memory
  → run_supervisor
  → END
```

Inside `run_supervisor`, the supervisor can delegate to worker agents.

---

## Worker agents

### Document Research Worker

Tools:

- `get_uploaded_documents`
- `search_documents`
- `summarize_document`
- `extract_paper_metadata`

Responsibilities:

- stay grounded in uploaded session documents
- answer only from document evidence
- avoid web assumptions
- return concise findings for the supervisor to synthesize

### Web Research Worker

Tools:

- `web_search`
- `fetch_webpage`

Responsibilities:

- gather current or external information
- use the web only when needed
- return concise findings rather than a final polished essay

### Computation Worker

Tools:

- `calculate`

Responsibilities:

- solve only the numerical subproblem
- provide exact calculations
- avoid speculative interpretation beyond the math

---

## Delegation model

The supervisor does **not** directly own the workers' low-level tools.

Instead, it gets three worker-facing delegation tools:

- `ask_document_worker(task)`
- `ask_web_worker(task)`
- `ask_computation_worker(task)`

Each of these tools internally runs a specialist worker agent with its own
restricted toolset.

This design is important because it:

- enforces specialization
- prevents tool sprawl at the supervisor level
- keeps worker behavior observable
- makes later evaluation and debugging easier

---

## Why workers are role-based, not infrastructure-based

The worker split is intentionally:

- document
- web
- math

and **not**:

- retriever
- reranker
- embedding
- memory

The rejected split mirrors backend implementation details rather than real
problem types. Agents work best when each role corresponds to a meaningful
research behavior, not an internal subsystem.

---

## Validation strategy

The current agent modes do **not** run the deep-mode validation loop.

This is deliberate:

- the agent path already adds latency through planning and tool use
- adding LLM-as-judge on top makes first-pass iteration slower and harder to
  debug
- we first want to observe natural agent behavior before introducing another
  reasoning layer

Validation can still be added later if runtime behavior shows that the extra
quality check is worth the cost.

---

## UI model

The frontend now groups chat execution into:

- `Workflows`
  - `Fast`
  - `Deep`
- `Agents`
  - `Single RAG Agent`
  - `Supervisor Agent`

This mirrors the backend contract and keeps the user-facing distinction clear:

- workflows = fixed, deterministic graphs
- agents = tool-using, adaptive reasoning systems

---

## Files involved

Key backend files:

- `src/agents/rag_agent.py`
- `src/agents/supervisor_agent.py`
- `src/agents/workers/document_research_agent.py`
- `src/agents/workers/web_research_agent.py`
- `src/agents/workers/computation_agent.py`
- `src/orchestrators/rag_agent_orchestrator.py`
- `src/orchestrators/supervisor_agent_orchestrator.py`
- `src/orchestrators/rag_orchestrator.py`
- `src/tools/`

Key frontend files:

- `ai_assistant_ui/src/store/chatStore.ts`
- `ai_assistant_ui/src/components/ChatArea.tsx`
- `ai_assistant_ui/src/types/api.ts`

---

## Summary

The project now has three distinct execution styles:

- `workflow / fast` — low-latency deterministic RAG
- `workflow / deep` — richer deterministic reasoning with clarification and validation
- `agent / single_rag_agent` — one flexible agent with multiple high-level tools
- `agent / supervisor_orchestration_agent` — one supervisor delegating to specialist document, web, and computation workers

The core design goal is to make agent behavior more flexible **without**
turning internal infrastructure into agent-facing tools.
