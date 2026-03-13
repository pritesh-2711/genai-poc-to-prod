# Integration Document: Frontend ↔ Backend

## Overview

The frontend is a React + TypeScript SPA built with Vite. The backend is a FastAPI
application served by Uvicorn. They communicate over HTTP/JSON using JWT Bearer tokens.

---

## Running the stack

### Backend

```bash
cd <project-root>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in DB credentials, JWT_SECRET_KEY
python api_server.py         # starts on http://localhost:8000
```

### Frontend

```bash
cd AI_Assistant_UI
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

## Chat message flow

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

The single POST endpoint handles the full cycle on the backend:
1. Persist user message
2. Fetch session history for LLM context
3. Call `ChatService.get_response_async`
4. Persist assistant reply
5. Return both records

The frontend never makes a separate GET for messages after sending — the POST
response contains both records, so the message list stays in sync without
a redundant fetch.

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

### Error shape

All 4xx/5xx responses return:

```json
{ "detail": "human-readable error message" }
```

The API client in `src/api/client.ts` reads `body.detail` and surfaces it via
store `error` state, which renders as a dismissible banner in `ChatArea`.

---

## State management

Two Zustand stores:

**`authStore`** — token, user profile, signin/signup/signout/loadMe actions.  
**`chatStore`** — sessions list, active session ID, messages, send/create/delete actions.

Both stores are module-level singletons. `ChatPage` resets `chatStore` on unmount
via `reset()` so stale session data is never shown after sign-out.

---

## Files changed / added

### Backend

| File | Change |
|------|--------|
| `src/core/models.py` | Added `terminated_at: Optional[datetime] = None` to `SessionRecord`. Removed unused `ChatMessage` dataclass. |
| `src/memory/repository.py` | All methods wrapped in `try/finally` to prevent connection leaks. `get_sessions` and `create_session` now select and populate `terminated_at`. |
| `src/api/chat.py` | `send_message` changed to `async def`; `asyncio.run()` replaced with direct `await`. Fixes 502 Bad Gateway on every request after the first. |
| `src/api/deps.py` | Removed `@lru_cache` + `Depends` anti-pattern. Singleton accessors now read from `app.state`. Per-request `get_repo` reads config from `app.state`. |
| `src/api/main.py` | Added `lifespan` context manager for startup/shutdown. `ConfigManager` and `ChatService` initialised once and stored on `app.state`. Logging initialised here. All Chainlit references removed. |
| `src/api/sessions.py` | `_to_session_response` reads `s.terminated_at` directly (no `getattr` workaround). |

### Frontend (new)

| File | Purpose |
|------|---------|
| `index.html` | Entry HTML, font imports |
| `vite.config.ts` | Vite config with `/api` proxy |
| `tsconfig.json` | TypeScript config |
| `package.json` | Dependencies |
| `src/main.tsx` | React root mount |
| `src/App.tsx` | Router, auth guard |
| `src/styles/global.css` | Design tokens (CSS vars), resets, markdown styles |
| `src/types/api.ts` | TypeScript types mirroring backend schemas |
| `src/api/client.ts` | All HTTP calls, token injection, error handling |
| `src/store/authStore.ts` | Zustand auth state |
| `src/store/chatStore.ts` | Zustand sessions + messages state |
| `src/pages/AuthPage.tsx` | Sign in / Sign up page |
| `src/pages/AuthPage.module.css` | Auth page styles |
| `src/pages/ChatPage.tsx` | Chat layout page |
| `src/pages/ChatPage.module.css` | Chat layout styles |
| `src/components/Sidebar.tsx` | Session list, create/delete/terminate, user info |
| `src/components/Sidebar.module.css` | Sidebar styles |
| `src/components/ChatArea.tsx` | Message list, input, sending state |
| `src/components/ChatArea.module.css` | Chat area styles |
| `src/components/MessageBubble.tsx` | Individual message with markdown rendering |
| `src/components/MessageBubble.module.css` | Message bubble styles |