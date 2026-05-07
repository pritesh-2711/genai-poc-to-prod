# Agent Architecture

This project supports two agentic chat variants alongside deterministic workflow modes:

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
  exact calculation, and data analysis in one answer.
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

The tools visible to the agent are high-level capabilities — each hides the
lower-level mechanics it needs. See the Tool split section below for how tools
are sourced.

### 3. Preserve deterministic memory resolution

Even in agent mode, short-term and long-term memory are resolved by the
orchestrator before the agent runs. Memory is still injected into the system
prompt in a controlled way; the agent is responsible for planning and
tool selection, not memory loading.

---

## Tool split — local vs MCP

Tools are sourced from two places.

### Local tools (session-scoped, live in `src/tools/`)

These tools need session context (user ID, session ID, embedder, database
repositories) that cannot be serialized over a network protocol.

| Tool | Description |
| --- | --- |
| `get_uploaded_documents` | List all documents ingested in this session |
| `search_documents` | Semantic search over session-ingested chunks |
| `summarize_document` | Generate a key-findings summary for a specific file |
| `extract_paper_metadata` | Pull author, abstract, and publication details |

### MCP tools (stateless, served by mcp-tools-library)

These tools have no session dependency and are served by the
[mcp-tools-library](https://github.com/pritesh-2711/mcp-tools-library)
MCP server. The application connects at startup via `MCPToolLoader`
(`src/mcp_client.py`) and loads all tools once into memory.

| Tool | Description |
| --- | --- |
| `calculate` | Safe arithmetic and math-module expression evaluator |
| `web_search` | Tavily-backed web search |
| `fetch_webpage` | Readable-text extraction from a URL |
| `analyse` | Runs pandas/Python code in an E2B cloud sandbox |
| `rav_idp_process_and_ingest` | RaV-IDP full extraction pipeline — extracts entities and writes JSON output |
| `rav_idp_get_document_fidelity` | RaV-IDP quick fidelity check — returns quality metrics only |

**Why the split?** Document tools are closures that close over the session
embedder and database repos. Moving them to a remote server would require
passing that context in every call, breaking isolation and adding significant
complexity. Stateless tools have no such constraint and benefit from being
deployed, versioned, and scaled independently.

---

## Single RAG Agent

`single_rag_agent` is one agent with access to all high-level tools — both
local document tools and all MCP tools.

**Best for:** ordinary research chat, document QA with light web augmentation,
data analysis questions where the user provides data inline, document entity
extraction questions, and smaller multi-step questions that do not need
specialist separation.

### Single RAG graph

```text
resolve_memory
  → run_agent
  → END
```

### Tool access

The single agent receives:

- all local document tools (4 tools)
- all MCP tools: `calculate`, `web_search`, `fetch_webpage`, `analyse`,
  `rav_idp_process_and_ingest`, `rav_idp_get_document_fidelity`

---

## Supervisor Orchestration Agent

### Purpose

`supervisor_orchestration_agent` is a multi-agent design:

- one supervisor
- five specialist workers

The supervisor plans the work, delegates to workers, and synthesizes the final
answer.

It is best for:

- document vs web comparisons
- multi-source research questions
- mixed evidence + calculation tasks
- questions requiring data analysis alongside document research
- structured entity extraction from documents

### Backend flow

```text
resolve_memory
  → run_supervisor
  → END
```

Inside `run_supervisor`, the supervisor can call any of its five delegation tools.

---

## Worker agents

### Document Research Worker

Tools: `get_uploaded_documents`, `search_documents`, `summarize_document`, `extract_paper_metadata`

Responsibilities:

- Stay grounded in uploaded session documents
- Answer only from document evidence; avoid web assumptions
- Return concise findings for the supervisor to synthesize

### Web Research Worker

Tools: `web_search`, `fetch_webpage`

Responsibilities:

- Gather current or external information
- Use focused search queries; fetch full pages only when snippets are insufficient
- Return concise findings rather than a polished essay

### Computation Worker

Tools: `calculate`

Responsibilities:

- Solve only the numerical subproblem
- Provide exact calculations; avoid speculation beyond the math

### Data Analysis Worker

Tools: `analyse`

Responsibilities:

- Perform quantitative analysis, EDA, or statistics using Python/pandas code
  executed in a secure E2B sandbox
- Work only with data explicitly provided in the task (CSV text, extracted tables,
  numbers from prior tool calls)
- Quote specific numbers from sandbox output; never fabricate results
- Keep analysis code focused — one analysis per tool call

Chart rules enforced in the worker prompt:

- Always call `plt.show()` — never `plt.savefig()`. The sandbox only returns
  display output; filesystem writes are not returned to the caller.
- Always set `plt.figure(figsize=(8, 5))` for consistent chart dimensions.
- When a chart is rendered, respond with one short sentence and at most two
  key insights — do not repeat every data point in text alongside the image.
- Charts are captured as base64 PNGs, validated in the MCP tool, threaded through
  `AgentRunResult.charts` → `RAGState.charts` → SSE `done` event, persisted in
  `orchestrator_metadata` JSONB, and returned on session reload.

### Document Extraction Worker

Tools: `rav_idp_process_and_ingest`, `rav_idp_get_document_fidelity`

Responsibilities:

- Extract structured entities from documents using the RaV-IDP pipeline
- Use `rav_idp_get_document_fidelity` for quick quality/confidence checks
  (does not write output to disk)
- Use `rav_idp_process_and_ingest` when the user needs the full entity set
  or when entities should be persisted for downstream use
- Report entity types, counts, fidelity scores, and low-confidence flags

---

## Delegation model

The supervisor does **not** directly own the workers' tools.

Instead, it gets five worker-facing delegation tools:

| Delegation tool | Routes to |
| --- | --- |
| `ask_document_worker(task)` | `DocumentResearchWorkerAgent` |
| `ask_web_worker(task)` | `WebResearchWorkerAgent` |
| `ask_computation_worker(task)` | `ComputationWorkerAgent` |
| `ask_data_analysis_worker(task)` | `DataAnalysisWorkerAgent` |
| `ask_document_extraction_worker(task)` | `DocumentExtractionWorkerAgent` |

Each delegation tool internally runs the specialist worker agent with its
restricted toolset and returns its findings as a string.

This design:

- enforces specialization — each worker only sees the tools relevant to its role
- prevents tool sprawl at the supervisor level
- keeps worker behavior observable and independently debuggable
- makes later per-worker evaluation straightforward

---

## Why workers are role-based, not infrastructure-based

The worker split is intentionally:

- document research
- web research
- computation
- data analysis
- document extraction

and **not**:

- retriever
- reranker
- embedding
- memory

The rejected split mirrors backend implementation details rather than real
problem types. Agents work best when each role corresponds to a meaningful
research behavior, not an internal subsystem.

---

## MCP client lifecycle

`MCPToolLoader` (`src/mcp_client.py`) manages the connection to the MCP server:

1. **Startup** — `lifespan` in `src/api/main.py` calls `await mcp_tool_loader.connect()`.
   This launches the MCP subprocess (stdio mode) or opens an HTTP session (streamable-http
   mode) and loads all tools into memory.
2. **Per-request** — Orchestrators call `mcp_tool_loader.get_tools(["web_search", ...])`.
   Tool objects are returned from the in-memory cache — no reconnection per request.
3. **Shutdown** — `lifespan` calls `await mcp_tool_loader.disconnect()`.

If the MCP server is unavailable at startup (wrong path, missing dependency), the app
still starts and tools degrade gracefully to an empty list. A warning is logged.

Transport is configured in `configs/config.yaml` under the `mcp:` key:

```yaml
mcp:
  enabled: true
  transport: "stdio"          # or "streamable-http" for a remote server
  stdio:
    command: "python"
    args: ["../mcp-tools-library/mcp_server.py"]
    env:
      TAVILY_API_KEY: "${TAVILY_API_KEY}"
      E2B_API_KEY: "${E2B_API_KEY}"
      RAV_IDP_MODE: "full"
  http:
    url: "${MCP_SERVER_URL}"
```

---

## Validation strategy

The current agent modes do **not** run the deep-mode validation loop.

This is deliberate:

- the agent path already adds latency through planning and tool use
- adding LLM-as-judge on top makes first-pass iteration slower and harder to debug
- we first want to observe natural agent behavior before introducing another reasoning layer

Validation can still be added later if runtime behavior shows that the extra quality
check is worth the cost.

---

## UI model

The frontend groups chat execution into:

- `Workflows` — `Fast`, `Deep`
- `Agents` — `Single RAG Agent`, `Supervisor Agent`

This mirrors the backend contract and keeps the user-facing distinction clear:

- workflows = fixed, deterministic graphs
- agents = tool-using, adaptive reasoning systems

---

## Files involved

### Agent core

- `src/agents/rag_agent.py` — `SingleRAGAgent`
- `src/agents/supervisor_agent.py` — `SupervisorOrchestrationAgent` (5 delegation tools)
- `src/agents/_shared.py` — `AgentRunResult` (includes `charts: list[str]`), `extract_agent_run_result`, `_extract_charts_from_messages`

### Diagram and visualisation utilities

- `src/orchestrators/mermaid_utils.py` — `is_valid_mermaid()` + `fix_mermaid_in_text()`:
  validates Mermaid blocks and attempts LLM self-correction before the response is persisted

### Worker agent files

- `src/agents/workers/document_research_agent.py`
- `src/agents/workers/web_research_agent.py`
- `src/agents/workers/computation_agent.py`
- `src/agents/workers/data_analysis_agent.py` ← new
- `src/agents/workers/document_extraction_agent.py` ← new

### Orchestrators

- `src/orchestrators/rag_agent_orchestrator.py`
- `src/orchestrators/supervisor_agent_orchestrator.py`
- `src/orchestrators/rag_orchestrator.py`
- `src/orchestrators/base.py` — `mcp_tool_loader` param added

### Tools

- `src/tools/document_tools.py` — local session-scoped tools (unchanged)
- `src/mcp_client.py` — `MCPToolLoader` (MCP server connection + tool cache)

### Config

- `src/core/models.py` — `MCPConfig` dataclass
- `src/core/config.py` — `_build_mcp_config()`
- `configs/config.yaml` — `mcp:` block

---

## Summary

The project now has three distinct execution styles:

| Mode | Variant | Description |
| --- | --- | --- |
| `workflow` | `fast` | Low-latency deterministic RAG |
| `workflow` | `deep` | Richer deterministic reasoning with clarification and validation |
| `agent` | `single_rag_agent` | One flexible agent with document + MCP tools |
| `agent` | `supervisor_orchestration_agent` | Supervisor + five specialist workers; document, web, math, data analysis, entity extraction |

The core design goal is to make agent behavior more flexible **without** turning
internal infrastructure into agent-facing tools, and to decouple stateless tool
implementations from the main application process via the MCP protocol.
