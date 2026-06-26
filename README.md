# Research Paper Chat Application

A production-oriented RAG chat application built step by step — from a bare LLM call to a multi-user API with memory, document ingestion, hierarchical retrieval, reranking, LangGraph-based workflows, and agentic orchestration.

## Quick Start

```bash
python api_server.py    # REST API (FastAPI + Uvicorn)
```

----

## Learning path for beginners

Each branch adds one capability on top of the previous. Follow them in order.

### Branch: `explore`

- Read `notebooks/explore_langchain.ipynb`
- Covers LangChain basics: prompt templates, chains, LLM providers (Ollama / OpenAI)

### Branch: `feature/beginners-app`

- Minimalistic project structure: logging, configs, Pydantic models, custom exceptions
- Chat capability with swappable LLM providers (Ollama / OpenAI)V
- Simple UI via Chainlit and a CLI chat system

### Branch: `feature/memory`

- Read `notebooks/explore_memory.ipynb` — validates the DB schema and the full conversation round-trip before any application code
- Read `docs/design_intuition.md` (Part 1) — explains why a three-table schema (users → sessions → chats) is the right design for multi-user isolation
- Read `docs/design_intuition.md` (Part 3) — explains the short-term / long-term memory split, activation guard, similarity threshold, and deduplication
- Run `sql/init.sql` against a local PostgreSQL instance (`poc_to_prod` database)
- Key additions in `src/memory/repository.py`: user auth (bcrypt), session management, conversation history retrieval
- Conversation history is split into two layers and injected into the LLM system prompt via `ChatService._build_system_prompt`:
  - **Short-term** — last `short_term_limit` messages (configurable, default 10), always included
  - **Long-term** — up to 10 semantically similar past messages retrieved via cosine search over `chats.embeddings`; only active once the session exceeds `short_term_limit` messages and only includes results above `long_term_similarity_threshold`

### Branch: `feature/compliance`

- Read `notebooks/explore_safety_evaluations.ipynb` — explores both NeMo Guardrails (Colang flows, topical rails, `self check` rails) and DeepEval safety metrics before wiring either into the app
- The notebook covers:
  - How Colang pattern-matching blocks harmful inputs without an extra LLM call
  - Why `self check input`/`self check output` rails require a tuned prompt and when they cause false positives
  - How DeepEval's `ToxicityMetric`, `BiasMetric`, and `GEval` can be used as pre-LLM input guards
- Application changes:
  - `src/guardrails/input_guard.py` — runs three DeepEval metrics concurrently (asyncio.gather) before every LLM call
  - `src/core/exceptions.py` — `InputBlockedError` distinguishes a blocked message from an LLM failure
  - `src/core/models.py` + `configs/config.yaml` — `GuardrailsConfig` controls which guards are active and which evaluator model to use
  - `src/api/chat.py` — blocked messages are saved as a polite assistant reply (HTTP 201) instead of surfacing as an error

### Branch: `feature/rag`

- Read `notebooks/explore_extraction.ipynb` — scans layout and then extracts text, tables and images from pdf files
- Read `notebooks/explore_chunking.ipynb` — explore and understand different types of chunking strategies
- Read `notebooks/explore_ingestion.ipynb` — validates every stage of the pipeline end-to-end before wiring into the application
- Read `docs/design_intuition.md` (Part 2) — explains the extraction pipeline, chunking strategy, embedding provider design, and vector schema decisions
- The notebook covers:
  - **Extraction** — `LayoutExtractor` (single Docling pass) → `TextExtractor` (filters text/latex records)
  - **Chunking** — three strategies explored (`HierarchicalChunker`, `TextTilingChunker`, `EmbeddingSemanticChunker`); hierarchical chosen for production
  - **Embeddings** — `LocalEmbedder`, `OllamaEmbedder`, `OpenAIEmbedder` all share `BaseEmbedder`; provider swapped via `configs/config.yaml` with no code changes
  - **Ingestion** — `chunk_with_parents()` → INSERT parents first, map int index to DB UUID, INSERT children with FK
  - **Retrieval** — embed query → `<=>` cosine search over `ingestions` → fetch parent chunks → pass to LLM
- Key schema additions in `sql/init.sql`:
  - `poc2prod.parenthierarchy` — large parent chunks (not vector-indexed; fetched by UUID)
  - `poc2prod.ingestions` — small child chunks with `VECTOR` embeddings (searched at query time)
  - `poc2prod.chats` — gains an `embeddings VECTOR` column; now actively written on every message for long-term memory search
  - `VECTOR` (no fixed dimension) used throughout — dimension enforced in application layer via `EmbeddingConfig`

### Branch: `feature/langgraph`

- Read `notebooks/explore_langgraph.ipynb` — explores LangGraph concepts: StateGraph, conditional edges, Send API (fan-out), `interrupt()` for HITL, and `MemorySaver` checkpointing
- Read `docs/design_intuition.md` (Part 4) — explains the Fast/Deep orchestrator design, reranker abstraction, and HITL pattern
- Application changes:
  - `src/orchestrators/` — LangGraph-based `FastOrchestrator` and `DeepOrchestrator`, composed by `RAGOrchestrator`
  - `src/reranker/` — `CrossEncoderReranker` (BGE) sitting behind `BaseReranker`, configurable via `configs/config.yaml`
  - `src/api/chat.py` — SSE streaming endpoint (`GET /sessions/{id}/stream`) with per-node status events in deep mode
  - UI — fast/deep mode toggle, node status shown next to typing indicator in deep mode

### Branch: `feature/agents`

- Read `docs/agents.md` — explains the `single_rag_agent` and `supervisor_orchestration_agent` designs, worker responsibilities, and why the tool boundaries are high-level rather than infrastructure-level
- Read `docs/design_intuition.md` (Part 5) — explains the reasoning behind deterministic workflows vs agentic orchestration, the supervisor/worker split, and the `category + variant` API contract
- Application changes:
  - `src/agents/` — `SingleRAGAgent`, `SupervisorOrchestrationAgent`, and specialist worker agents for document research, web research, and computation
  - `src/orchestrators/` — `RAGAgentOrchestrator` and `SupervisorAgentOrchestrator`, both routed by the top-level `RAGOrchestrator`
  - `src/tools/` — high-level tools exposed to agents (`search_documents`, `web_search`, `fetch_webpage`, `calculate`, etc.)
  - `src/api/schemas.py` + UI — chat execution now uses `category` (`workflow` | `agent`) and `variant` (`fast`, `deep`, `single_rag_agent`, `supervisor_orchestration_agent`)

### Branch: `feature/intersession-feedback` (current)

- Read `docs/design_intuition.md` (Part 6) — explains intersession memory design, RLHF-lite weighted retrieval, and the Laplace-smoothed chunk scoring formula
- Application changes:
  - `src/databases/intersession.py` — `IntersessionRepository` (asyncpg): stores per-session LLM-generated summaries with pgvector embeddings; retrieves top-K semantically similar prior-session summaries at query time; Laplace-smoothed chunk score recomputation
  - `src/jobs/` — APScheduler background jobs: `run_intersession_memory_job` (nightly, summarises sessions via LLM, embeds summaries); `run_chunk_scoring_job` (weekly, recomputes Laplace scores from raw feedback counts)
  - `src/orchestrators/state.py` — `RAGState` gains `intersession_context: str` and `retrieved_chunk_ids: list[str]`
  - `src/orchestrators/base.py` — `_resolve_memory_node` fetches and injects intersession summaries; `_rerank_and_build_context_node` records the child chunk UUIDs that were used
  - `src/chat_service.py` — `_build_system_prompt` injects intersession summaries between RAG context and long-term memory
  - `src/databases/retrieval.py` — `PgVectorRetrievalRepository.search()` LEFT JOINs `chunk_scores` and ranks by `(1-α)·cosine + α·quality_score` (RLHF-weighted)
  - `src/memory/repository.py` — `save_feedback()` and `attribute_feedback_to_chunks()` for the new `feedback` and `chunk_scores` tables
  - `src/api/chat.py` — `POST /sessions/{id}/messages/{chat_id}/feedback` endpoint; `retrieved_chunk_ids` stored in `orchestrator_metadata`
  - `src/api/schemas.py` — `FeedbackRequest`, `FeedbackResponse`
  - `src/core/models.py` + `configs/config.yaml` — `IntersessionConfig`, `ChunkScoringConfig`, `JobsConfig` dataclasses; `jobs:` YAML block
  - `sql/init.sql` — three new tables: `session_summaries`, `feedback`, `chunk_scores`
  - `requirements.txt` — `apscheduler>=3.10.4`
  - Frontend — thumbs up/down `FeedbackBar` on every assistant message bubble; `feedbackState` in Zustand `chatStore` with optimistic update + rollback; `submitFeedback` in `chatApi`; `FeedbackRequest`/`FeedbackResponse` types

### Branch: `mcp`

- Read `docs/agents.md` — updated with the two new workers and the full tool split between local (session-scoped document tools) and MCP (stateless utility, analysis, and extraction tools)
- Stateless tools (`web_search`, `fetch_webpage`, `calculate`) moved out of the application process and into a dedicated MCP server: [mcp-tools-library](https://github.com/pritesh-2711/mcp-tools-library)
- Two new tools introduced via the MCP server:
  - **`analyse`** — runs pandas/Python code in an E2B cloud sandbox; no host filesystem exposure
  - **`rav_idp_process_and_ingest` / `rav_idp_get_document_fidelity`** — document entity extraction with per-entity fidelity scoring (RaV-IDP pipeline)
- Two new specialist worker agents introduced:
  - **`DataAnalysisWorkerAgent`** — owns the `analyse` tool; handles quantitative EDA, statistics, and trend questions
  - **`DocumentExtractionWorkerAgent`** — owns the RaV-IDP tools; handles entity extraction and document quality assessment
- Application changes:
  - `src/mcp_client.py` — `MCPToolLoader`: connects to the MCP server at startup (stdio subprocess or HTTP), loads all tools once, exposes them by name to orchestrators
  - `src/agents/workers/` — two new workers added (`DataAnalysisWorkerAgent`, `DocumentExtractionWorkerAgent`)
  - `src/agents/supervisor_agent.py` — two new delegation tools (`ask_data_analysis_worker`, `ask_document_extraction_worker`) on the supervisor
  - `src/orchestrators/` — `BaseOrchestrator` and `RAGOrchestrator` accept `mcp_tool_loader`; agent orchestrators load tools from MCP at request time
  - `src/tools/` — only session-scoped document tools remain local; stateless tools removed
  - `src/core/models.py` + `configs/config.yaml` — `MCPConfig` controls transport (stdio or HTTP), server path/URL, and env vars forwarded to the subprocess

#### Visualisation and diagram support (current work)

Charts from the `analyse` tool (E2B PNG output) and Mermaid flow diagrams from
the LLM are now first-class response objects — rendered in the UI, persisted
across sessions, and downloadable.

#### Chart pipeline (`analyse` tool → PNG → UI)

- `mcp-tools-library/server/tools/analysis.py` — `_extract_and_validate_charts()` scans
  E2B `result.results` for `.png` attributes, validates each against PNG magic bytes
  (`\x89PNG`), and includes valid images in the tool response under the `charts` key.
- `src/agents/_shared.py` — `AgentRunResult` gains a `charts: list[str]` field.
  `_extract_charts_from_messages()` scans LangChain `ToolMessage` objects for the
  `charts` key, handling both plain-string and list-of-content-block content formats.
- `src/orchestrators/state.py` — `RAGState` includes `charts: list[str]` in its output section.
- `src/api/chat.py` — charts are stored in `orchestrator_metadata` JSONB alongside the
  assistant message and returned in the SSE `done` event and non-streaming response.
- `src/memory/repository.py` — `get_conversation_history` now SELECTs `orchestrator_metadata`
  and extracts `charts` from it so chart images survive page refresh and session reload.
- `src/api/schemas.py` — `ChatMessageResponse` includes `charts: list[str] = []`.
- `src/core/models.py` — `ChatRecord` includes `charts: list = []`.

#### Mermaid diagram pipeline (LLM text → rendered SVG)

- `src/orchestrators/mermaid_utils.py` — new module with two helpers:
  - `is_valid_mermaid(code)` — checks first non-blank non-comment line against all
    known Mermaid diagram type keywords; also validates `subgraph`/`end` count parity
    and rejects HTML entities that break Mermaid.js rendering.
  - `fix_mermaid_in_text(text, chat_service)` — async; finds all ` ```mermaid ``` `
    blocks, validates each, attempts one LLM self-correction pass for invalid blocks,
    replaces unfixable blocks with empty string (silent removal).
- `src/api/chat.py` — calls `fix_mermaid_in_text` before persisting or streaming the
  assistant response, in both the streaming and non-streaming paths.
- `src/chat_service.py` — `_MERMAID_RULES` constant appended to every `_build_system_prompt`
  call, giving all execution modes (workflow fast/deep, both agent variants) consistent
  instructions: use Mermaid for structural diagrams, use `analyse` + matplotlib for data
  charts, never output raw SVG, plain ASCII labels only, every `subgraph` needs `end`.

----

## What the initial version was lacking

- Query is sent directly to the LLM without understanding intent or complexity
- No conversation history — only the current message is used as context
- Responses are limited to the LLM's training knowledge (no document grounding)
- Harmful content and jailbreaking are not handled
- No way to evaluate response quality
- Not built for multiple users
- Anyone who signs up can immediately access the system

## Checklist we are solving throughout this repo

- Query Analysis
- Memory: short-term, long-term, intersession, user feedbacks & preferences
- Feedback Learning
- RAG
- Guardrails
- Evaluations
- Tool calling
- Workflows & Agents
- A good system design

----

## What has been covered

- [x] **Memory** — PostgreSQL-backed conversation history per user per session. Split into short-term (last N messages, bounded by `short_term_limit`) and long-term (cosine similarity search over embedded chat history, gated by session length and `long_term_similarity_threshold`). Duplicates between layers are removed before the system prompt is assembled.
- [x] **Multi-user support** — JWT-authenticated REST API. Each user sees only their own sessions and messages.
- [x] **Guardrails** — DeepEval metrics (`ToxicityMetric`, `BiasMetric`, `GEval`) run concurrently before every LLM call. Blocked messages return a friendly assistant reply, not an error. Configurable via `guardrails:` block in `configs/config.yaml`.
- [x] **RAG** — PDF/DOCX upload → extraction → hierarchical chunking → embedding → pgvector storage → cosine retrieval → parent-document context → grounded LLM response. Fully session-scoped.
- [x] **Reranker** — Cross-encoder (`BAAI/bge-reranker-base`) re-scores retrieved chunks before passing context to the LLM. Configurable model, `top_k`, and device via `reranker:` block in `configs/config.yaml`.
- [x] **LangGraph orchestration** — Two execution modes selectable per request:
  - **Fast mode** — resolve memory → retrieve → rerank → generate. No extra LLM calls. Optimised for latency.
  - **Deep mode** — intent analysis → optional HITL clarification (via `interrupt()`) → complexity routing → query rewrite or decomposition (Send API fan-out) → retrieve → rerank → generate → LLM-as-judge validation loop (max 3 iterations, best-response fallback).
- [x] **Agentic orchestration** — Two agent variants selectable per request:
  - **Single RAG Agent** — one agent with access to all high-level document, web, calculation, data analysis, and extraction tools.
  - **Supervisor Orchestration Agent** — one supervisor delegating to five specialist workers (document research, web research, computation, data analysis, document extraction) via worker-facing delegation tools.
- [x] **MCP tools library** — Stateless tools (`web_search`, `fetch_webpage`, `calculate`) and new tools (`analyse`, `rav_idp_*`) served from a dedicated MCP server ([mcp-tools-library](https://github.com/pritesh-2711/mcp-tools-library)). The application connects via `langchain-mcp-adapters` at startup and keeps the connection alive. Session-scoped document tools remain local.
- [x] **Chart and diagram rendering** — The `analyse` tool returns E2B-generated PNG charts; charts are validated, threaded through the agent and orchestrator pipeline, persisted in `orchestrator_metadata` JSONB, and served back on session reload. LLM responses containing Mermaid code blocks are validated server-side (keyword check + subgraph/end parity + HTML entity check), corrected via one LLM retry if invalid, and rendered as interactive SVGs in the frontend. Both charts and diagrams support copy-to-clipboard and download from the UI.
- [x] **Intersession memory** — Background job (nightly by default, configurable via `jobs.intersession.summary_interval_hours`) summarises each user session using the LLM, embeds the summary with pgvector, and stores it in `session_summaries`. At query time, the `_resolve_memory_node` fetches the top-K most semantically similar prior-session summaries (cosine search over the user's own summaries, excluding the current session) and injects them into the system prompt between RAG context and long-term memory. Total injected text is capped at `intersession_context_max_tokens` (configurable). Token count is approximated as `len(text) / 4`.
- [x] **Feedback Learning (RLHF-lite)** — Thumbs up/down rating UI on every persisted assistant message. `POST /sessions/{id}/messages/{chat_id}/feedback` stores the rating in the `feedback` table and synchronously increments positive/negative counters in `chunk_scores` for the chunks that contributed to that response (extracted from `orchestrator_metadata.retrieved_chunk_ids`). A weekly background job recomputes each chunk's quality score using Laplace smoothing: `score = (positive + 1) / (positive + negative + 2)`. Retrieval ranking blends cosine similarity with the quality score: `(1 - α) × cosine + α × chunk_score`, where `α = rlhf_alpha` (default 0.2, configurable). Chunks with no feedback default to 0.5 (neutral — not boosted or penalised).
- [x] **SSE streaming** — Token-level streaming via `GET /sessions/{id}/stream`. Deep mode also emits `status` events naming the current node (e.g., "Checking query intent…", "Ranking relevant results…").
- [x] **Unified execution contract** — Chat requests use `category + variant` so workflows and agents share one API surface:
  - `workflow / fast`
  - `workflow / deep`
  - `agent / single_rag_agent`
  - `agent / supervisor_orchestration_agent`
- [x] **Admin-gated signup** — New registrations land as `status='pending'`. Users cannot sign in until an admin sets their status to `'approved'` via a direct SQL update. Rejected users receive a clear message on sign-in attempt.

## Still to address

- [ ] Post-LLM Evaluations (response quality, hallucination, relevance)
- [ ] True token-level streaming (replace word-split with `stream_mode="messages"`)
- [ ] Show decomposed sub-queries to user before retrieval runs
- [ ] Agent evaluation / analytics (compare workflow vs agent quality, cost, and latency)

----

## Admin: approving users

New signups are stored with `status = 'pending'`. To approve or reject a user, run a direct SQL query against the database:

```sql
-- Approve a user
UPDATE poc2prod.users SET status = 'approved' WHERE email = 'user@example.com';

-- Reject a user
UPDATE poc2prod.users SET status = 'rejected' WHERE email = 'user@example.com';

-- List all pending requests
SELECT user_id, name, email, created_at FROM poc2prod.users WHERE status = 'pending' ORDER BY created_at;
```

----

## License

See LICENSE file for details.
