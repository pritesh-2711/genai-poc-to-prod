# Research Paper Chat Application

A production-oriented RAG chat application built step by step — from a bare LLM call to a multi-user API with memory, document ingestion, hierarchical retrieval, reranking, and LangGraph-based orchestration.

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
- [x] **SSE streaming** — Token-level streaming via `GET /sessions/{id}/stream`. Deep mode also emits `status` events naming the current node (e.g., "Checking query intent…", "Ranking relevant results…").
- [x] **Admin-gated signup** — New registrations land as `status='pending'`. Users cannot sign in until an admin sets their status to `'approved'` via a direct SQL update. Rejected users receive a clear message on sign-in attempt.

## Still to address

- [ ] Feedback Learning
- [ ] Post-LLM Evaluations (response quality, hallucination, relevance)
- [ ] Tool calling
- [ ] True token-level streaming (replace word-split with `stream_mode="messages"`)
- [ ] Show decomposed sub-queries to user before retrieval runs

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
