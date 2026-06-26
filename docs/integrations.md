# Integration Document: Frontend - Backend - MCP Tools

## Overview

The frontend is a React + TypeScript SPA built with Vite. The backend is a FastAPI
application served by Uvicorn. They communicate over HTTP/JSON using JWT Bearer tokens,
with a secondary SSE channel for streaming responses.

The system is a session-scoped RAG assistant with two execution modes and two agent modes:

- **Fast mode** — direct retrieval + reranking + generation (low latency)
- **Deep mode** — intent analysis, optional clarification, query decomposition, retrieval, reranking, generation, and LLM-as-judge validation
- **Single RAG Agent** — one agent with access to all document and MCP tools
- **Supervisor Agent** — supervisor + five specialist workers

The backend connects to a separate **MCP tools server**
([mcp-tools-library](https://github.com/pritesh-2711/mcp-tools-library))
at startup. The MCP server provides stateless tools (`web_search`, `fetch_webpage`,
`calculate`, `analyse`, `rav_idp_*`) that agents use at request time.

---

## Running the stack

### MCP tools server (required for agent modes)

The MCP server is launched automatically as a subprocess when the backend starts
(stdio transport, the default). Its Python dependencies must be installed into the
**same virtual environment** as the backend, or the subprocess will fail to import.

```bash
# From the repo root — install MCP server deps into the backend venv
cd mcp
source .venv/bin/activate
pip install -e ../mcp-tools-library/
pip install -e "../mcp-tools-library/[rav-idp]"   # optional — full entity extraction
```

To run the MCP server as a standalone process instead (HTTP transport):

```bash
cd mcp-tools-library
python mcp_server.py
```

Then switch `configs/config.yaml` to `transport: "streamable-http"` and set
`MCP_SERVER_URL` in your `.env`.

### Backend

```bash
cd mcp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in DB credentials, JWT_SECRET_KEY, OPENAI_API_KEY,
                             # TAVILY_API_KEY, E2B_API_KEY
python api_server.py         # starts on http://localhost:8000
```

On startup you will see:

```text
MCP tools loaded (6 tools): ['calculate', 'web_search', 'fetch_webpage',
                              'analyse', 'rav_idp_process_and_ingest',
                              'rav_idp_get_document_fidelity']
Application startup complete. LLM=openai/gpt-4.1-mini, ..., MCP=stdio
Intersession memory job scheduled (interval=24h)
Chunk scoring job scheduled (interval=168h)
```

If MCP fails to connect the app still starts — agent modes run with only local
document tools.

### Frontend

```bash
cd ai_assistant_ui
npm install
npm run dev                  # starts on http://localhost:5173
```

---

## Proxy configuration

Vite proxies all `/api/*` requests to `http://localhost:8000`, stripping the `/api`
prefix before forwarding. This means:

- Frontend calls `/api/auth/signin`
- Vite rewrites to `http://localhost:8000/auth/signin`
- No CORS issues during development

In `vite.config.ts`:

```ts
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, ''),
  },
}
```

In production, configure your reverse proxy (nginx, Caddy, etc.) to do the same
rewrite, or update `BASE_URL` in `src/api/client.ts` to point directly at the API.

---

## Authentication flow

### Sign-up (admin-gated)

```text
User submits name + email + password
  → POST /api/auth/signup         (SignUpRequest)
  ← { message: "...", status: "pending" }   201

Frontend shows "awaiting admin approval" screen.
No token is issued. User cannot sign in until approved.
```

The admin approves the user by running:

```sql
UPDATE poc2prod.users SET status = 'approved' WHERE email = 'user@example.com';
```

If a user tries to sign up with an already-registered email, the backend returns `409 Conflict`.

### Sign-in

```text
User submits email + password
  → POST /api/auth/signin         (SignInRequest)

  ← 200 { access_token, token_type }   if approved
  ← 401 "Invalid email or password."  if wrong credentials
  ← 403 "Your account is awaiting admin approval."   if pending
  ← 403 "Your account access has been declined."     if rejected
```

On success:

```text
Store token in localStorage
  → GET /api/auth/me              (Bearer <token>)
  ← { user_id, name, email, created_at }

Token stored; user redirected to /chat
```

All subsequent requests attach the token as:

```text
Authorization: Bearer <access_token>
```

Token expiry is 7 days (configured in `deps.py` via `ACCESS_TOKEN_EXPIRE_DAYS`).
On 401 from any request, the frontend clears the token and redirects to `/auth`.

---

## Session management flow

On app load after authentication:

```text
GET /api/sessions
  ← SessionResponse[]   (newest first)

If active sessions exist → auto-select most recent active session
  GET /api/sessions/{session_id}/messages
  ← ChatMessageResponse[]

If no sessions exist → show empty state; user creates session explicitly
```

The frontend never auto-creates sessions on load. Session creation is an explicit
user action. This prevents the double-POST race condition observed when creation
was triggered automatically during startup.

### Session lifecycle

| Action          | API call                                      | Effect                              |
|-----------------|-----------------------------------------------|-------------------------------------|
| Create          | `POST /sessions`                              | `is_active=true`                    |
| Select          | `GET /sessions/{id}/messages`                 | Load history into chat view         |
| End session     | `POST /sessions/{id}/terminate`               | `is_active=false`, stamps timestamp |
| Delete session  | `DELETE /sessions/{id}`                       | Hard delete with CASCADE on chats   |

---

## Document upload flow

```text
User clicks the upload button (left of the chat textarea)
  → POST /api/sessions/{session_id}/upload
      multipart/form-data: file, file_description (optional)
  ← UploadResponse {
       session_id, filename, file_path, size_bytes,
       content_type, file_description,
       parent_chunks, child_chunks
     }
```

Backend pipeline on upload (synchronous, returns when complete):

1. Verify session ownership + active status
2. Validate file type (PDF or DOCX only, max 50 MB)
3. Save to `storage/{user_id}/active/{session_id}/{filename}`
4. Run `IngestionPipeline.run()`:
   - **Extract**: `LayoutExtractor` → `TextExtractor` (text/latex records) + `TableExtractor` (markdown table records)
   - **Chunk**: `HierarchicalChunker.chunk_with_parents()` — parent ≈ 2000 chars, children ≈ 400 chars. Text records and table records chunked independently with running parent-index offsets to build a global parent→UUID map
   - **Embed**: `OllamaEmbedder.embed()` with `"search_document: "` prefix for all child chunks
   - **Ingest**: `PgVectorIngestionRepository.ingest_documents()` — parents to `parenthierarchy`, children+embeddings to `ingestions`
5. Return chunk counts

Frontend state after upload:

- `documentsStore.uploadFile()` tracks per-file status: `uploading → processing → done | error`
- On completion, calls `loadDocuments()` to refresh the session document list
- `PersonalDrive` panel auto-opens during upload

---

## Chat message flow (SSE streaming)

```text
User types message, selects mode (fast/deep), presses Enter
  → GET /api/sessions/{session_id}/stream?message=...&mode=fast|deep
      (EventSource / fetch with ReadableStream)
```

### SSE event types

| Event type     | When emitted                                   | Payload (`content`)           |
|----------------|------------------------------------------------|-------------------------------|
| `user_message` | Immediately after persisting the user message  | The user's message text       |
| `status`       | Deep mode only — on each new LangGraph node    | Human-readable node label     |
| `token`        | As LLM output accumulates                      | Partial response text         |
| `clarification`| Deep mode — when graph needs user input        | Question to show the user     |
| `done`         | After all tokens emitted                       | —                             |
| `error`        | On any error                                   | Error message                 |

### Status labels by node (deep mode only)

| Node                     | Status label shown                  |
|--------------------------|-------------------------------------|
| `resolve_memory`         | Loading conversation history…       |
| `analyze_query`          | Checking query intent…              |
| `route_complexity`       | Identifying complexity…             |
| `query_rewrite`          | Optimising query for retrieval…     |
| `query_decompose`        | Breaking down your question…        |
| `retrieve`               | Searching documents…                |
| `retrieve_sub_query`     | Searching documents…                |
| `rerank_and_build_context` | Ranking relevant results…         |
| `generate`               | Generating response…                |
| `validate_response`      | Validating answer quality…          |
| `correction`             | Refining the answer…                |

Status events are deduplicated — if three parallel `retrieve_sub_query` nodes run, "Searching documents…" is emitted only once. `query_clarification` and `finalize` emit no status.

### HITL clarification flow

When deep mode determines the query is unclear:

```text
SSE: { type: "clarification", content: "Could you clarify...?" }
User types reply
  → POST /api/sessions/{session_id}/stream/clarify
      body: { message: "...", session_id: "..." }
  ← Resumes SSE stream from paused graph state
```

The backend stores `session_id → thread_id` in `app.state.pending_clarifications`. The resume endpoint retrieves the thread ID and calls `Command(resume=user_reply)` on the `MemorySaver`-backed graph.

### Backend RAG cycle per message

1. Persist user message
2. **Orchestrate** via `RAGOrchestrator.astream_updates()`:
   - **Fast**: resolve memory → embed query → cosine search (`top_k=10`) → cross-encoder rerank (`top_k=5`) → assemble context → generate
   - **Deep**: (see design_intuition.md Part 4 for the full graph)
3. Persist assistant reply with `orchestrator_metadata` (mode, intent, complexity, iteration count)
4. SSE stream ends with `done` event

Retrieval is best-effort — if no documents are uploaded, the LLM responds from its own knowledge.

---

## Document listing flow

```text
On session select (ChatPage useEffect):
  → GET /api/sessions/{session_id}/documents
  ← DocumentRecord[] {
       filename, file_description, file_type,
       parent_chunks, child_chunks, ingested_at
     }
```

Results are grouped by filename from `poc2prod.ingestions` using `COUNT(DISTINCT parent_id)`.

---

## Database schema

### `users`

| Column | Type | Notes |
| --- | --- | --- |
| user_id | uuid PK | |
| name | varchar(255) | |
| email | varchar(255) | unique |
| password | text | bcrypt hash |
| status | varchar(20) | `'pending'` \| `'approved'` \| `'rejected'` — default `'pending'` |
| created_at | timestamptz | |
| updated_at | timestamptz | |
| last_login_at | timestamptz | updated on every successful sign-in |

### `sessions`

| Column | Type | Notes |
| --- | --- | --- |
| session_id | uuid PK | |
| user_id | uuid FK | ON DELETE CASCADE |
| session_name | varchar(60) | |
| is_active | boolean | |
| created_at | timestamptz | |
| terminated_at | timestamptz | nullable |

### `chats`

| Column | Type | Notes |
| --- | --- | --- |
| chat_id | uuid PK | |
| session_id | uuid FK | ON DELETE CASCADE |
| sender | text | `'user'` \| `'assistant'` |
| message | text | |
| embeddings | vector | pgvector; used for long-term memory search |
| orchestrator_metadata | jsonb | mode, intent, complexity, iteration_count, `retrieved_chunk_ids` (list of child chunk UUIDs used in final RAG context), charts (base64 PNGs), etc. |
| created_at | timestamptz | |

### `parenthierarchy` — large parent chunks (≈2000 chars), not embedded

| Column | Type | Notes |
| --- | --- | --- |
| id | uuid PK | |
| parent_chunk_content | text | |
| filename | text | |
| metadata | jsonb | `{ page, type, bbox? }` |
| content_type | varchar(20) | `'text'` \| `'table'` \| `'image'` |

### `ingestions` — child chunks with embeddings

| Column | Type | Notes |
| --- | --- | --- |
| id | uuid PK | |
| parent_id | uuid FK | references parenthierarchy |
| user_id | uuid | |
| session_id | uuid | |
| filename | text | |
| file_description | text | |
| type | varchar | `'pdf'` \| `'doc'` |
| chunk_content | text | child chunk text |
| embeddings | vector | pgvector |
| metadata | jsonb | `{ page, type, bbox? }` |
| content_type | varchar(20) | `'text'` \| `'table'` \| `'image'` |
| created_at | timestamptz | |

### `session_summaries` — intersession memory

| Column | Type | Notes |
| --- | --- | --- |
| id | uuid PK | |
| user_id | uuid FK | ON DELETE CASCADE |
| session_id | uuid FK | UNIQUE; ON DELETE CASCADE |
| summary_text | text | LLM-generated summary of the session |
| summary_embedding | vector | pgvector; used for cosine similarity lookup |
| token_count | int | approximate token count (`len(summary) // 4`) |
| created_at | timestamptz | |
| updated_at | timestamptz | |

Populated by the nightly `run_intersession_memory_job`. Upserted on `session_id` conflict so re-runs update existing summaries.

### `feedback` — thumbs up / down ratings

| Column | Type | Notes |
| --- | --- | --- |
| id | uuid PK | |
| chat_id | uuid FK | ON DELETE CASCADE; references `chats` |
| session_id | uuid FK | ON DELETE CASCADE; references `sessions` |
| user_id | uuid FK | ON DELETE CASCADE; references `users` |
| rating | varchar(4) | `'up'` \| `'down'` — CHECK constraint |
| comment | text | optional free-text comment |
| created_at | timestamptz | |

UNIQUE on `(user_id, chat_id)` — upserted so a user can change their rating.

### `chunk_scores` — RLHF quality scores per chunk

| Column | Type | Notes |
| --- | --- | --- |
| chunk_id | uuid PK | FK → `ingestions(id)` ON DELETE CASCADE |
| positive_count | int | cumulative thumbs-up count |
| negative_count | int | cumulative thumbs-down count |
| score | float | Laplace-smoothed quality score: `(pos+1)/(pos+neg+2)` |
| updated_at | timestamptz | |

Default `score = 0.5` (neutral). Updated synchronously on feedback submit (counter increment) and recomputed in batch by the weekly `run_chunk_scoring_job`.

---

## Embedding: asymmetric model

The default embedder is `OllamaEmbedder` using `nomic-embed-text-v2-moe:latest`.

This is an asymmetric retrieval model that requires different task prefixes:
- **Indexing** (`embed()` / `embed_one()`): prepends `"search_document: "`
- **Query** (`embed_query()`): prepends `"search_query: "`

Symmetric models (OpenAI, local) use `embed_query()` falling back to `embed_one()` — no prefix needed.

---

## API contract reference

All types in `src/types/api.ts` mirror `src/api/schemas.py` field-for-field.

### Auth endpoints

| Method | Path            | Request body    | Response                     |
|--------|-----------------|-----------------|------------------------------|
| POST   | /auth/signup    | SignUpRequest   | SignUpResponse 201           |
| POST   | /auth/signin    | SignInRequest   | TokenResponse 200 / 401 / 403 |
| POST   | /auth/signout   | —               | 204                          |
| GET    | /auth/me        | —               | UserResponse 200             |

**`SignUpResponse`**
```json
{ "message": "Registration successful. Your account is awaiting admin approval.", "status": "pending" }
```

### Session endpoints

| Method | Path                           | Request body          | Response               |
|--------|--------------------------------|-----------------------|------------------------|
| GET    | /sessions                      | —                     | SessionResponse[] 200  |
| POST   | /sessions                      | CreateSessionRequest  | SessionResponse 201    |
| DELETE | /sessions/{id}                 | —                     | 204                    |
| POST   | /sessions/{id}/terminate       | —                     | 204                    |

### Chat / stream endpoints

| Method | Path                             | Request body / params               | Response                      |
|--------|----------------------------------|-------------------------------------|-------------------------------|
| GET    | /sessions/{id}/messages          | —                                   | ChatMessageResponse[] 200     |
| POST   | /sessions/{id}/messages          | SendMessageRequest                  | SendMessageResponse 201       |
| GET    | /sessions/{id}/stream            | `?message=...&mode=fast\|deep`      | SSE stream                    |
| POST   | /sessions/{id}/stream/clarify    | `{ message, session_id }`           | SSE stream (resumed)          |

**`SendMessageRequest`**
```json
{ "message": "...", "mode": "fast" }
```

### Feedback endpoint

| Method | Path                                            | Request body      | Response               |
|--------|-------------------------------------------------|-------------------|------------------------|
| POST   | /sessions/{id}/messages/{chat_id}/feedback      | FeedbackRequest   | FeedbackResponse 201   |

**`FeedbackRequest`**
```json
{ "rating": "up" | "down", "comment": "optional free text" }
```

**`FeedbackResponse`**
```json
{ "feedback_id": "uuid", "chat_id": "uuid", "session_id": "uuid", "rating": "up" }
```

Re-submitting feedback for the same `(user_id, chat_id)` pair upserts the row — the rating can be changed.
On submit, `retrieved_chunk_ids` from `orchestrator_metadata` are read synchronously to increment the correct counters in `chunk_scores`.

### Document endpoints

| Method | Path                           | Request body / form          | Response                   |
|--------|--------------------------------|------------------------------|----------------------------|
| POST   | /sessions/{id}/upload          | multipart: file, description | UploadResponse 201         |
| GET    | /sessions/{id}/documents       | —                            | DocumentRecord[] 200       |

### Error shape

All 4xx/5xx responses return:

```json
{ "detail": "human-readable error message" }
```

The API client in `src/api/client.ts` reads `body.detail` and surfaces it via
store `error` state, which renders as a dismissible banner in `ChatArea`.

---

## State management

Three Zustand stores:

**`authStore`** — token, user profile, signin/signup/signout/loadMe actions.

**`chatStore`** — sessions list, active session ID, messages, send/create/delete actions. Gains `statusContent: string | null` for deep mode node status, cleared on `done`/`error`/`clarification` events. Also holds `feedbackState: Record<string, 'up' | 'down'>` (keyed by `chat_id`) updated optimistically on feedback submit, rolled back on API failure.

**`documentsStore`** — per-session document lists, upload queue with status tracking (`uploading → processing → done | error`), drive panel open/close state.

All stores are module-level singletons. `ChatPage` resets `chatStore` on unmount
via `reset()` so stale session data is never shown after sign-out.

---

## Files changed / added

### Backend — original integration

| File | Change |
|------|--------|
| `src/core/models.py` | Added `terminated_at: Optional[datetime] = None` to `SessionRecord`. Removed unused `ChatMessage` dataclass. |
| `src/memory/repository.py` | All methods wrapped in `try/finally` to prevent connection leaks. `get_sessions` and `create_session` now select and populate `terminated_at`. |
| `src/api/chat.py` | `send_message` changed to `async def`; `asyncio.run()` replaced with direct `await`. Fixes 502 Bad Gateway on every request after the first. |
| `src/api/deps.py` | Removed `@lru_cache` + `Depends` anti-pattern. Singleton accessors now read from `app.state`. Per-request `get_repo` reads config from `app.state`. |
| `src/api/main.py` | Added `lifespan` context manager for startup/shutdown. `ConfigManager` and `ChatService` initialised once and stored on `app.state`. Logging initialised here. All Chainlit references removed. |
| `src/api/sessions.py` | `_to_session_response` reads `s.terminated_at` directly (no `getattr` workaround). |

### Backend — RAG + document ingestion

| File | Change |
|------|--------|
| `src/api/upload.py` | New. `POST /sessions/{id}/upload` — validates, saves, runs `IngestionPipeline`, returns chunk counts. |
| `src/api/documents.py` | New. `GET /sessions/{id}/documents` — groups `ingestions` by filename, returns aggregate chunk counts. |
| `src/api/chat.py` | Added RAG: `embed_query()` + vector search + parent context fetch + co-located retrieval. `_RAG_TOP_K=10`, `_MAX_CONTEXT_CHARS=8000`. |
| `src/api/loader.py` | New. `FileLoader.save()` — writes uploaded bytes to `storage/{user_id}/active/{session_id}/{filename}`. |
| `src/databases/pipeline.py` | New. `IngestionPipeline` orchestrates extract→chunk→embed→ingest. Integrates `TableExtractor`. Fixes parent-id offset bug. |
| `src/databases/ingestion.py` | New. `PgVectorIngestionRepository.ingest_documents()` — inserts parents then children with embeddings. Accepts `content_type` param. |
| `src/databases/retrieval.py` | New. `PgVectorRetrievalRepository` with `search()`, `fetch_parent_contexts()`, and `fetch_colocated_chunks()`. |
| `src/embedding/base.py` | Added `embed_query()` with default fallback to `embed_one()`. |
| `src/embedding/ollama.py` | Added nomic model auto-detection, `doc_prefix`/`query_prefix` params, asymmetric prefixes. |
| `src/chunker/hierarchical.py` | `chunk_with_parents()` — parent 2000 chars, child 400 chars. |
| `src/extraction/table.py` | `TableExtractor` — extracts tables as markdown via Docling. |
| `sql/migration_001_content_type.sql` | Adds `content_type VARCHAR(20)` to `ingestions` and `parenthierarchy`. Applied. |

### Backend — LangGraph orchestration + reranker

| File | Change |
|------|--------|
| `src/orchestrators/state.py` | New. `RAGState(TypedDict)` — full graph state including `raw_chunks: Annotated[list[dict], operator.add]` for fan-out accumulation. `SubQueryState` for parallel sub-query nodes. |
| `src/orchestrators/base.py` | New. `BaseOrchestrator(ABC)` — shared nodes (`resolve_memory`, `retrieve`, `rerank_and_build_context`, `generate`). |
| `src/orchestrators/fast_orchestrator.py` | New. `FastOrchestrator` — linear 4-node graph. |
| `src/orchestrators/deep_orchestrator.py` | New. `DeepOrchestrator` — full deep graph with HITL, fan-out, validation loop. |
| `src/orchestrators/rag_orchestrator.py` | New. `RAGOrchestrator` — top-level router graph; wraps Fast + Deep; exposes `ainvoke`, `aresume`, `astream_updates`, `get_graph_state`, `is_interrupted`, `get_clarification_question`. Uses `MemorySaver`. |
| `src/reranker/base.py` | New. `BaseReranker(ABC)` with abstract `rerank(query, chunks, top_k)`. |
| `src/reranker/cross_encoder.py` | New. `CrossEncoderReranker` — sentence-transformers `CrossEncoder`, adds `rerank_score` key to each chunk. |
| `src/api/chat.py` | Rewritten. `stream_message` uses `astream_updates(subgraphs=True)` for per-node status SSE. `_DEEP_NODE_STATUS` maps node names to labels. Clarification resume via `POST .../stream/clarify`. |
| `src/api/main.py` | Lifespan initialises `CrossEncoderReranker` + `RAGOrchestrator` on `app.state`. Adds `app.state.pending_clarifications: dict[str, str]`. |
| `src/api/schemas.py` | `SendMessageRequest` gains `mode: Literal["fast", "deep"] = "fast"`. |
| `src/core/models.py` | Added `RerankerConfig` dataclass. |
| `src/core/config.py` | `ConfigManager._build_reranker_config()` reads `reranker:` block. |
| `configs/config.yaml` | Added `reranker:` block (model, top_k, device). |
| `requirements.txt` | Added `sentence-transformers>=3.0.0`. |
| `sql/init.sql` | `chats` table gains `orchestrator_metadata JSONB DEFAULT '{}'::jsonb`. |
| `sql/migration_user_approval.sql` (applied) | Adds `status VARCHAR(20)` column to `users`; sets pre-existing users to `'approved'`. |

### Backend — user access control

| File | Change |
|------|--------|
| `src/memory/repository.py` | Added `UserNotApprovedError(account_status)`. `authenticate_user()` fetches `status` column; raises `UserNotApprovedError` for non-approved accounts after password verification. |
| `src/api/auth.py` | `signup` returns `SignUpResponse` (message + status) instead of `UserResponse`. `signin` catches `UserNotApprovedError` → 403 with status-specific message. |
| `src/api/schemas.py` | Added `SignUpResponse(message, status)`. |
| `sql/init.sql` | `users` table gains `status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (...)`. |

### Frontend — original

| File | Purpose |
|------|---------|
| `index.html` | Entry HTML, font imports |
| `vite.config.ts` | Vite config with `/api` proxy |
| `tsconfig.json` | TypeScript config |
| `package.json` | Dependencies (Vite 7, React 19, Tailwind 3) |
| `src/main.tsx` | React root mount |
| `src/App.tsx` | Router, auth guard |
| `src/styles/global.css` | Design tokens (CSS vars), resets, markdown styles |
| `src/index.css` | Tailwind CSS v3 directives |
| `src/types/api.ts` | TypeScript types mirroring backend schemas |
| `src/api/client.ts` | All HTTP calls, token injection, error handling |
| `src/store/authStore.ts` | Zustand auth state |
| `src/store/chatStore.ts` | Zustand sessions + messages state |
| `src/pages/SignIn.tsx` | Sign in page |
| `src/pages/SignUp.tsx` | Sign up page |
| `src/pages/ChatPage.tsx` | Chat layout page |
| `src/components/Sidebar.tsx` | Session list, create/delete/terminate, user info, drive toggle |
| `src/components/ChatArea.tsx` | Message list, input bar, upload button, fast/deep toggle, node status indicator |
| `src/components/ChatArea.module.css` | Chat area styles including `.modeToggle`, `.thinkingStatus` (with `fadeIn`) |
| `src/components/MessageBubble.tsx` | Individual message with markdown rendering |

### Frontend — document drive

| File | Purpose |
|------|---------|
| `src/store/documentsStore.ts` | Zustand store: per-session doc lists, upload queue |
| `src/components/drive/PersonalDrive.tsx` | Slide-in panel showing ingested documents and upload progress |

### Frontend — LangGraph mode + streaming

| File | Change |
|------|--------|
| `src/types/api.ts` | `SendMessageRequest` gains `mode?: 'fast' \| 'deep'` |
| `src/api/client.ts` | `StreamEvent` union gains `{ type: 'status'; content: string }` and `{ type: 'clarification'; content: string }` |
| `src/store/chatStore.ts` | Gains `statusContent: string \| null`; set on `status` events; cleared on `done`/`error`/`clarification`. `sendMessage(text, mode)` passes `mode` to the stream endpoint. |
| `src/components/ChatArea.tsx` | `mode` local state (`'fast'` default). Toggle button in hint row. Shows `statusContent` next to three dots in deep mode (with `key={statusContent}` for per-change fade-in). |
| `src/components/ChatArea.module.css` | Added `.hintRow`, `.modeToggle`, `.modeToggleDeep`, `.modeDot`, `.thinkingStatus` |

### Frontend — user access control

| File | Change |
|------|--------|
| `src/pages/SignUp.tsx` | Removed auto sign-in after signup. Shows "Request submitted" confirmation screen with `CheckCircle2` icon and link back to sign-in. |
| `src/services/api.ts` | `signUp()` return type changed from `Promise<User>` to `Promise<{ message: string; status: string }>`. |

### Backend — intersession memory + RLHF-lite feedback

| File | Change |
|------|--------|
| `sql/init.sql` | Three new tables appended: `session_summaries` (intersession memory with pgvector), `feedback` (thumbs up/down ratings), `chunk_scores` (RLHF quality counters + Laplace score). |
| `src/core/models.py` | Added `IntersessionConfig`, `ChunkScoringConfig`, `JobsConfig` dataclasses. |
| `src/core/config.py` | Added `_build_jobs_config()` method; `self.jobs_config` populated in `__init__`. |
| `configs/config.yaml` | Added `jobs:` block (`intersession.enabled`, `summary_interval_hours`, `max_summaries_per_prompt`, `intersession_context_max_tokens`; `chunk_scoring.interval_hours`, `rlhf_alpha`). |
| `requirements.txt` | Added `apscheduler>=3.10.4`. |
| `src/databases/intersession.py` | New. `IntersessionRepository` (asyncpg): `upsert_session_summary`, `get_relevant_summaries` (cosine search, excludes current session), `get_sessions_for_summary`, `get_session_chat_history_text`, `recompute_chunk_scores` (Laplace batch update). |
| `src/jobs/__init__.py` | New. Empty package init. |
| `src/jobs/intersession_memory.py` | New. `run_intersession_memory_job` — iterates all sessions, summarises dialogue via LLM (max 16 000 chars of history), embeds summary, upserts to `session_summaries`. |
| `src/jobs/chunk_scoring.py` | New. `run_chunk_scoring_job` — delegates to `IntersessionRepository.recompute_chunk_scores()`. |
| `src/jobs/scheduler.py` | New. `create_scheduler()` factory: builds `AsyncIOScheduler` with interval jobs wired to config; returns scheduler (not yet started). |
| `src/orchestrators/state.py` | Added `intersession_context: str` and `retrieved_chunk_ids: list[str]` to `RAGState`. |
| `src/orchestrators/base.py` | `_resolve_memory_node` fetches top-K intersession summaries and builds truncated `intersession_context` string. `_rerank_and_build_context_node` collects `retrieved_chunk_ids` from reranked chunks. `_generate_node` passes `intersession_context` to `chat_service.get_response_async()`. |
| `src/orchestrators/rag_orchestrator.py` | Accepts `intersession_repo` and `intersession_config` params; passes them through `shared_kwargs`. |
| `src/chat_service.py` | `get_response_async`, `stream_response_async`, `_build_system_prompt` accept `intersession_context: Optional[str]`; injected between RAG context and long-term memory in the system prompt. |
| `src/databases/retrieval.py` | `PgVectorRetrievalRepository.__init__` accepts `rlhf_alpha`. `search()` LEFT JOINs `chunk_scores`; ORDER BY `(1-alpha)*cosine + alpha*COALESCE(score, 0.5) DESC`. |
| `src/memory/repository.py` | Added `save_feedback()` (INSERT/ON CONFLICT upsert to `feedback`) and `attribute_feedback_to_chunks()` (reads `orchestrator_metadata.retrieved_chunk_ids`, updates `chunk_scores` counters). |
| `src/api/schemas.py` | Added `FeedbackRequest` and `FeedbackResponse`. |
| `src/api/chat.py` | Added `POST /sessions/{id}/messages/{chat_id}/feedback` endpoint. Stores `retrieved_chunk_ids` in `orchestrator_metadata` for both streaming and non-streaming paths. |
| `src/api/main.py` | Lifespan creates `IntersessionRepository`, passes `rlhf_alpha` to `PgVectorRetrievalRepository`, wires `intersession_repo` + `intersession_config` to `RAGOrchestrator`, creates and starts APScheduler. |

### Frontend — feedback UI

| File | Change |
|------|--------|
| `src/types/api.ts` | Added `FeedbackRequest` and `FeedbackResponse` interfaces. |
| `src/api/client.ts` | Added `chatApi.submitFeedback(sessionId, chatId, body)` — `POST /sessions/{sessionId}/messages/{chatId}/feedback`. |
| `src/store/chatStore.ts` | Added `feedbackState: Record<string, FeedbackRating>` state field; `submitFeedback(sessionId, chatId, rating, comment?)` action with optimistic update and rollback on failure. `reset()` clears `feedbackState`. |
| `src/components/MessageBubble.tsx` | Added `FeedbackBar` inline component (thumbs up/down icon buttons with green/red highlight on selection). Rendered below charts on persisted assistant messages only (requires `chatId` + `sessionId` props). Added `chatId?: string` and `sessionId?: string` to component props. |
| `src/components/ChatArea.tsx` | Passes `chatId` and `sessionId` to `MessageBubble` for assistant messages. |
