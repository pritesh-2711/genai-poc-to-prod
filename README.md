# Research Paper Chat Application

A production-oriented chat application built step by step — from a bare LLM call to a multi-user API with memory, document ingestion, RAG retrieval, and guardrails.

## Quick Start

```bash
python main.py          # CLI mode
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
- Chat capability with swappable LLM providers (Ollama / OpenAI)
- Simple UI via Chainlit and a CLI chat system

### Branch: `feature/memory`

- Read `notebooks/explore_memory.ipynb` — validates the DB schema and the full conversation round-trip before any application code
- Read `docs/design_intuition.md` (Part 1) — explains why a three-table schema (users → sessions → chats) is the right design for multi-user isolation
- Run `sql/init.sql` against a local PostgreSQL instance (`poc_to_prod` database)
- Key additions in `src/memory/repository.py`: user auth (bcrypt), session management, conversation history retrieval
- Conversation history is injected into the LLM system prompt on every call via `ChatService._build_system_prompt`

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
  - `poc2prod.chats` — gains an `embeddings VECTOR` column for future semantic history search
  - `VECTOR` (no fixed dimension) used throughout — dimension enforced in application layer via `EmbeddingConfig`
- Application changes:
  - `src/databases/pipeline.py` — `IngestionPipeline.run()` orchestrates extract → chunk → embed → ingest in one async call
  - `src/databases/ingestion.py` — `PgVectorIngestionRepository` (asyncpg) inserts parent + child rows
  - `src/databases/retrieval.py` — `PgVectorRetrievalRepository` (asyncpg) performs cosine search and parent context fetch
  - `src/api/upload.py` — accepts PDF/DOCX, saves to `storage/{user_id}/active/{session_id}/`, runs pipeline, returns chunk counts
  - `src/api/chat.py` — embeds query, retrieves top-K session-scoped chunks, fetches parent contexts, injects RAG block into system prompt
  - `src/chat_service.py` — `get_response_async()` accepts `rag_context` injected before conversation history in the system prompt
  - `src/api/loader.py` — storage layout is `storage/{user_id}/active|archive/{session_id}/`

----

## What the initial version was lacking

- Query is sent directly to the LLM without understanding intent or complexity
- No conversation history — only the current message is used as context
- Responses are limited to the LLM's training knowledge (no document grounding)
- Harmful content and jailbreaking are not handled
- No way to evaluate response quality
- Not built for multiple users

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

- [x] **Memory** — PostgreSQL-backed conversation history per user per session. History injected as context into every LLM call.
- [x] **Multi-user support** — JWT-authenticated REST API. Each user sees only their own sessions and messages.
- [x] **Guardrails** — DeepEval metrics (`ToxicityMetric`, `BiasMetric`, `GEval`) run concurrently before every LLM call. Blocked messages return a friendly assistant reply, not an error.
- [x] **RAG** — PDF/DOCX upload → extraction → hierarchical chunking → embedding → pgvector storage → cosine retrieval → parent-document context → grounded LLM response. Fully session-scoped.

## Still to address

- [ ] Query Analysis / Intent detection
- [ ] Feedback Learning
- [ ] Post-LLM Evaluations (response quality, hallucination, relevance)
- [ ] Tool calling
- [ ] Workflows
- [ ] Agents

----

## License

See LICENSE file for details.
