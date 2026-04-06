# Integration Document: Frontend ↔ Backend

## Overview

The frontend is a React + TypeScript SPA built with Vite. The backend is a FastAPI
application served by Uvicorn. They communicate over HTTP/JSON using JWT Bearer tokens.

The system is a session-scoped RAG (Retrieval-Augmented Generation) assistant:
users upload documents into a session, and every chat message retrieves relevant
passages from those documents before calling the LLM.

---

## Running the stack

### Backend

```bash
cd main
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in DB credentials, JWT_SECRET_KEY, OPENAI_API_KEY
python api_server.py         # starts on http://localhost:8000
```

Ollama must be running with the embedding model pulled:

```bash
ollama serve
ollama pull nomic-embed-text-v2-moe
```

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

```
User submits email + password
  → POST /api/auth/signin         (SignInRequest)
  ← { access_token, token_type }

Store token in localStorage
  → GET /api/auth/me              (Bearer <token>)
  ← { user_id, name, email, created_at }

Token stored; user redirected to /
```

All subsequent requests attach the token as:

```
Authorization: Bearer <access_token>
```

Token expiry is 7 days (configured in `deps.py` via `ACCESS_TOKEN_EXPIRE_DAYS`).
On 401 from any request, the frontend clears the token and redirects to `/auth`.

Sign-up flow auto-signs-in after account creation:

```
POST /api/auth/signup   → 201 UserResponse
POST /api/auth/signin   → 200 TokenResponse
GET  /api/auth/me       → 200 UserResponse
```

---

## Session management flow

On app load after authentication:

```
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

```
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
   - **Ingest**: `PgVectorIngestionRepository.ingest_documents()` — parents to `parenthierarchy`, children+embeddings to `ingestions`. Text chunks use `content_type='text'`, table chunks use `content_type='table'`
5. Return chunk counts

Frontend state after upload:

- `documentsStore.uploadFile()` tracks per-file status: `uploading → processing → done | error`
- On completion, calls `loadDocuments()` to refresh the session document list
- `PersonalDrive` panel auto-opens during upload

---

## Chat message flow (RAG)

```
User types message, presses Enter
  → POST /api/sessions/{session_id}/messages
      body: { message: "..." }
  ← {
       user_message:      ChatMessageResponse,
       assistant_message: ChatMessageResponse
     }

Both messages appended to local state immediately on response.
```

Backend RAG cycle for each message:

1. Persist user message
2. **Embed query**: `OllamaEmbedder.embed_query()` with `"search_query: "` prefix (asymmetric retrieval)
3. **Vector search**: cosine similarity over `ingestions` WHERE `session_id = X`, top-10 child chunks
4. **Fetch parent contexts**: collect unique `parent_id` UUIDs → fetch full parent text from `parenthierarchy`
5. **Co-located retrieval**: extract page numbers + filenames from parent metadata → `fetch_colocated_chunks()` returns any table/image chunks from the same pages (regardless of vector score), appended to context as `[Table p.N]` / `[Image p.N]`
6. Build RAG context block (capped at 8000 chars) and inject into system prompt
7. Fetch conversation history, call `ChatService.get_response_async` (LLM)
8. Persist assistant reply
9. Return both records

Retrieval is best-effort — if no documents are uploaded or vector search returns nothing, the LLM responds from its own knowledge.

The frontend never makes a separate GET for messages after sending — the POST response contains both records.

---

## Document listing flow

```
On session select (ChatPage useEffect):
  → GET /api/sessions/{session_id}/documents
  ← DocumentRecord[] {
       filename, file_description, file_type,
       parent_chunks, child_chunks, ingested_at
     }
```

Results are grouped by filename from `poc2prod.ingestions` using `COUNT(DISTINCT parent_id)`.
No separate tracking table — ingestion counts are derived on-the-fly from existing data.

---

## Database schema notes

Two core tables in the `poc2prod` schema:

**`parenthierarchy`** — large parent chunks (≈2000 chars), not embedded

| Column | Type | Notes |
| --- | --- | --- |
| id | uuid PK | |
| parent_chunk_content | text | |
| filename | text | |
| metadata | jsonb | `{ page, type, bbox? }` |
| content_type | varchar(20) | `'text'` \| `'table'` \| `'image'` |

**`ingestions`** — child chunks with embeddings

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

The `content_type` column was added via `main/sql/migration_001_content_type.sql`.

---

## Embedding: asymmetric model

The default embedder is `OllamaEmbedder` using `nomic-embed-text-v2-moe:latest`.

This is an asymmetric retrieval model that requires different task prefixes:
- **Indexing** (`embed()` / `embed_one()`): prepends `"search_document: "`
- **Query** (`embed_query()`): prepends `"search_query: "`

Symmetric models (OpenAI, local) use `embed_query()` falling back to `embed_one()` — no prefix needed. The `BaseEmbedder.embed_query()` default delegates to `embed_one()` so symmetric models require no changes.

---

## API contract reference

All types in `src/types/api.ts` mirror `src/api/schemas.py` field-for-field.

### Auth endpoints

| Method | Path            | Request body    | Response           |
|--------|-----------------|-----------------|--------------------|
| POST   | /auth/signup    | SignUpRequest   | UserResponse 201   |
| POST   | /auth/signin    | SignInRequest   | TokenResponse 200  |
| POST   | /auth/signout   | —               | 204                |
| GET    | /auth/me        | —               | UserResponse 200   |

### Session endpoints

| Method | Path                           | Request body          | Response               |
|--------|--------------------------------|-----------------------|------------------------|
| GET    | /sessions                      | —                     | SessionResponse[] 200  |
| POST   | /sessions                      | CreateSessionRequest  | SessionResponse 201    |
| DELETE | /sessions/{id}                 | —                     | 204                    |
| POST   | /sessions/{id}/terminate       | —                     | 204                    |

### Chat endpoints

| Method | Path                           | Request body       | Response                  |
|--------|--------------------------------|--------------------|---------------------------|
| GET    | /sessions/{id}/messages        | —                  | ChatMessageResponse[] 200 |
| POST   | /sessions/{id}/messages        | SendMessageRequest | SendMessageResponse 201   |

### Document endpoints

| Method | Path                           | Request body / form         | Response                   |
|--------|--------------------------------|-----------------------------|----------------------------|
| POST   | /sessions/{id}/upload          | multipart: file, description | UploadResponse 201         |
| GET    | /sessions/{id}/documents       | —                           | DocumentRecord[] 200       |

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

**`chatStore`** — sessions list, active session ID, messages, send/create/delete actions.

**`documentsStore`** — per-session document lists, upload queue with status tracking (`uploading → processing → done | error`), drive panel open/close state.

All stores are module-level singletons. `ChatPage` resets `chatStore` on unmount
via `reset()` so stale session data is never shown after sign-out.

When a session is selected in `ChatPage`, `loadDocuments(sessionId)` is called
automatically to populate the drive panel.

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
| `src/api/documents.py` | New. `GET /sessions/{id}/documents` — groups `ingestions` by filename, returns aggregate chunk counts. No JOIN to `parenthierarchy` (avoids cross-session inflation). |
| `src/api/chat.py` | Added RAG: `embed_query()` + vector search + parent context fetch + co-located retrieval. `_RAG_TOP_K=10`, `_MAX_CONTEXT_CHARS=8000`. |
| `src/api/loader.py` | New. `FileLoader.save()` — writes uploaded bytes to `storage/{user_id}/active/{session_id}/{filename}`. |
| `src/databases/pipeline.py` | New. `IngestionPipeline` orchestrates extract→chunk→embed→ingest. Integrates `TableExtractor`. Fixes parent-id offset bug (running offset ensures children across multiple records map to correct parent UUIDs). |
| `src/databases/ingestion.py` | New. `PgVectorIngestionRepository.ingest_documents()` — inserts parents then children with embeddings. Accepts `content_type` param. |
| `src/databases/retrieval.py` | New. `PgVectorRetrievalRepository` with `search()`, `fetch_parent_contexts()`, and `fetch_colocated_chunks()` (page+filename-based co-location query). |
| `src/embedding/base.py` | Added `embed_query()` with default fallback to `embed_one()`. Documents the asymmetric model pattern. |
| `src/embedding/ollama.py` | Added nomic model auto-detection (`_NOMIC_MODELS`), `doc_prefix`/`query_prefix` params, asymmetric prefixes applied in `embed()` and `embed_query()`. |
| `src/chunker/hierarchical.py` | `HierarchicalChunker.chunk_with_parents()` — parent 2000 chars, child 400 chars. Local parent indices (0-based per call) must be offset by the pipeline before extending combined lists. |
| `src/extraction/table.py` | `TableExtractor` — uses Docling to extract tables as markdown. |
| `src/extraction/layout.py` | `LayoutExtractor` — Docling document layout parse. |
| `src/extraction/text.py` | `TextExtractor` — yields text/latex records from layout. |
| `sql/migration_001_content_type.sql` | New. Adds `content_type VARCHAR(20)` to both `ingestions` and `parenthierarchy`. Applied to live DB. |
| `.gitignore` | Added `data/` and `storage/` to prevent large files from being committed. |

### Frontend — original

| File | Purpose |
|------|---------|
| `index.html` | Entry HTML, font imports |
| `vite.config.ts` | Vite config with `/api` proxy |
| `tsconfig.json` | TypeScript config |
| `package.json` | Dependencies (Vite 7, React 19, Tailwind 3) |
| `src/main.tsx` | React root mount; imports both `global.css` and `index.css` |
| `src/App.tsx` | Router, auth guard |
| `src/styles/global.css` | Design tokens (CSS vars), resets, markdown styles |
| `src/index.css` | Tailwind CSS v3 directives |
| `src/types/api.ts` | TypeScript types mirroring backend schemas |
| `src/api/client.ts` | All HTTP calls, token injection, error handling |
| `src/store/authStore.ts` | Zustand auth state |
| `src/store/chatStore.ts` | Zustand sessions + messages state |
| `src/pages/AuthPage.tsx` | Sign in / Sign up page |
| `src/pages/AuthPage.module.css` | Auth page styles |
| `src/pages/ChatPage.tsx` | Chat layout page; wires `PersonalDrive` + `loadDocuments` on session select |
| `src/pages/ChatPage.module.css` | Chat layout styles |
| `src/components/Sidebar.tsx` | Session list, create/delete/terminate, user info, drive toggle button |
| `src/components/Sidebar.module.css` | Sidebar styles including `.driveBtn` / `.driveBtnActive` |
| `src/components/ChatArea.tsx` | Message list, always-visible input bar (disabled when no session), upload button wired to `documentsStore.uploadFile()` |
| `src/components/ChatArea.module.css` | Chat area styles including `.inputAreaDisabled` |
| `src/components/MessageBubble.tsx` | Individual message with markdown rendering |
| `src/components/MessageBubble.module.css` | Message bubble styles |

### Frontend — document drive

| File | Purpose |
|------|---------|
| `src/store/documentsStore.ts` | Zustand store: per-session doc lists, upload queue with status, drive open/close |
| `src/components/drive/PersonalDrive.tsx` | Slide-in panel showing ingested documents and upload progress per session |
| `postcss.config.js` | Updated to `{ tailwindcss: {}, autoprefixer: {} }` for Tailwind v3 |
