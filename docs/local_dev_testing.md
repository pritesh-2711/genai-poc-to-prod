# Local Dev Testing with Docker

## Prerequisites

These system packages must be on the host machine to run the frontend dev server (backend runs fully inside Docker):

```bash
node >= 18
npm >= 9
```

---

## Step 1 — Create your `.env`

```bash
cd pilot/
cp .env.example .env
```

Fill in `.env`:

```env
OPENAI_API_KEY=sk-...
JWT_SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
DB_HOST=localhost        # compose overrides this to "postgres" inside the stack
DB_PORT=5432
DB_NAME=poc2prod
DB_USER=postgres          # use postgres superuser — see note below
DB_PASSWORD=yourpassword
```

> **Note on DB_USER**: The `pgvector/pgvector:pg16` image creates a superuser from
> `POSTGRES_USER`. Use `postgres` (or whatever you set) consistently in both the `.env`
> and when connecting via `psql`. If you see `role "poc2prod_user" does not exist`,
> connect as `postgres` instead.

Leave all AWS vars blank — `storage.deployment: local` so they are never read at runtime.

---

## Step 2 — Build and start the backend stack

```bash
docker compose up --build
```

On the **first build** Docker will:
- Pull `pgvector/pgvector:pg16` and run `sql/init.sql` (schema, extensions, indexes)
- Build the backend image — this takes a while the first time because it:
  - Installs all system packages (`poppler-utils`, `tesseract-ocr`, `libreoffice`, etc.)
  - Downloads CPU-only torch (~250 MB) from `download.pytorch.org/whl/cpu`
  - Installs all Python packages from `requirements.txt`
  - Pre-downloads the RapidOCR ONNX model weights into the image
- Starts the backend with `--reload` (code changes hot-reload without rebuild)

Backend available at `http://localhost:8000`.

> **If the build times out** during a large pip download (network blip), just re-run
> `docker compose up --build` — Docker caches completed layers and resumes.

---

## Step 3 — Start the frontend

In a separate terminal:

```bash
cd ai_assistant_ui/
npm install
npm run dev
```

Frontend at `http://localhost:5173`. Vite proxies `/api/*` → `http://localhost:8000/*` so there are no CORS issues in local dev.

---

## Step 4 — Sign up and approve your account

1. Open `http://localhost:5173` and sign up.
2. The account lands in `status = 'pending'` — sign-in returns 403 until approved.
3. Approve via SQL:

```bash
docker exec -it poc2prod_postgres psql -U postgres -d poc2prod
```

```sql
UPDATE poc2prod.users SET status = 'approved' WHERE email = 'you@example.com';

-- verify
SELECT email, status FROM poc2prod.users;
```

---

## Code changes made to support Docker deployment

The following changes were required after cloning `pilot/` from `main/`. They are already committed — this section documents *why* each change was made.

### 1. `Dockerfile` (new file)

Two-stage build:
- **Builder stage**: installs Python packages using `python:3.12-slim` + build tools (`build-essential`, `libpq-dev`, `libmagic-dev`)
- **Runtime stage**: copies installed packages; installs all required system libraries:

| Package | Reason |
|---|---|
| `libpq5` | psycopg2 runtime |
| `libxcb1`, `libgl1`, `libglib2.0-0` | OpenCV headless runtime |
| `libgomp1` | GNU OpenMP — required by ONNX Runtime / layout models |
| `libmagic1` | `python-magic` file type detection |
| `poppler-utils` | PDF rendering (`pdftotext`, `pdftoppm`) |
| `tesseract-ocr`, `tesseract-ocr-eng`, `tesseract-ocr-hin` | Tesseract OCR engine |
| `pandoc`, `ghostscript`, `unrtf` | Document format conversion |
| `libreoffice` | DOCX/PPTX/XLSX processing via Unstructured |

Additional Dockerfile decisions:
- **CPU-only torch**: torch and torchvision are installed from `download.pytorch.org/whl/cpu` before `requirements.txt` to avoid the 1.5 GB CUDA build. GPU is not needed — the LLM runs on OpenAI, and the reranker/embedder run fine on CPU.
- **RapidOCR models pre-downloaded**: `RapidOCR()` is called during build as root. Without this, RapidOCR tries to download its ONNX weights to `site-packages/` at runtime and hits permission denied (container runs as non-root `appuser`).
- **`opencv-python-headless` forced**: `easyocr` pulls in `opencv-python` (full, with X11 GUI deps) as its dependency. A `--force-reinstall opencv-python-headless` step after the main install ensures the headless version wins.
- **Non-root user**: container runs as `appuser` (uid 1000) for security.

### 2. `docker-compose.yml` (new file)

Local dev stack: `postgres` + `backend` (no Redis — `storage.deployment: local`).

- Uses `pgvector/pgvector:pg16` which has the `vector` extension pre-installed.
- Mounts the source directory into the container so `--reload` picks up code changes without rebuild.
- `DB_HOST` is overridden to `postgres` (the service name) inside compose.

### 3. `requirements.txt`

| Change | Reason |
|---|---|
| Removed `torch`, `torchvision` | Moved to Dockerfile pre-install step with CPU wheel index |
| Removed `opencv-python` (non-headless) | Causes `libxcb.so.1: No such file or directory` inside container; headless variant is sufficient |
| Added `onnxruntime>=1.19.0` | RapidOCR depends on it but it was missing from the file; without it Docling reports "No OCR engine found" |

### 4. `src/extraction/layout.py` — disable picture classification

```python
# Before
def __init__(self, do_picture_classification: bool = True) -> None:

# After
def __init__(self, do_picture_classification: bool = False) -> None:
```

**Why**: with `do_picture_classification=True`, Docling's `DocumentFigureClassifier` triggers PyTorch Inductor JIT compilation which requires a C++ compiler (`g++`) at runtime. The runtime Docker image is `python:3.12-slim` (no compiler). The picture classifier labels figures as "chart / photo / diagram" — metadata we don't use anywhere in chunking, embedding, or retrieval. Disabling it also removes the `DocumentFigureClassifier` model download on every upload (faster cold starts).

### 5. `src/api/chat.py` — `InputBlockedError` in streaming endpoint

The non-streaming `send_message` endpoint already caught `InputBlockedError` and returned a polite blocked reply. The streaming `event_generator` did not — the error propagated through LangGraph and was logged as "Unexpected error". Added a specific `except InputBlockedError` block that persists the blocked reply as an assistant message and emits it as a proper SSE `done` event.

### 6. `src/guardrails/input_guard.py` — lower GEval threshold

```python
# Before
threshold=0.5

# After
threshold=0.3
```

GEval was false-positiving on short greetings like "Hi" (scored below 0.5 because they aren't clearly "a legitimate question or request"). Lowering to 0.3 means only inputs that score strongly as malicious are blocked — actual injection/jailbreak attempts score near 0 so detection still works.

### 7. `ai_assistant_ui/Dockerfile` (new file)

Two-stage build: Node 20 builds the Vite bundle; nginx 1.27 serves the static files.

- `VITE_API_BASE_URL` build arg bakes the backend URL into the bundle. Required for production (nginx has no proxy); in local dev the Vite dev server handles proxying so this is left empty.
- `nginx.conf` handles SPA routing (`try_files $uri /index.html`) and sets 1-year cache on static assets (safe because Vite uses content-hashed filenames).

### 8. `ai_assistant_ui/src/services/api.ts`

```typescript
// Before
const BASE_URL = '/api';

// After
const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '') + '/api';
```

In local dev `VITE_API_BASE_URL` is undefined so `BASE_URL` stays `/api` (Vite proxy handles it). In production the Docker build arg sets the full backend URL.

---

## Useful commands

```bash
# Stop everything (keeps volumes / data)
docker compose down

# Wipe DB and start fresh (drops all data)
docker compose down -v

# View all logs live
docker compose logs -f

# Backend logs only
docker compose logs -f backend

# Rebuild backend after requirements.txt or Dockerfile change
docker compose build --no-cache backend && docker compose up -d

# Rebuild backend (uses layer cache — faster if only Python code changed)
docker compose up --build backend

# Connect to the database
docker exec -it poc2prod_postgres psql -U postgres -d poc2prod
```

---

## Known build behaviours (not errors)

| Message | Meaning |
|---|---|
| `Warning: You are sending unauthenticated requests to the HF Hub` | HuggingFace rate-limits unauthenticated pulls. Set `HF_TOKEN` in `.env` if builds are slow. |
| `UNEXPECTED: roberta.embeddings.position_ids` | Expected when loading `BAAI/bge-reranker-base` from a different task architecture. Safe to ignore. |
| `torch_dtype is deprecated! Use dtype instead!` | Upstream Docling warning, not our code. |
| `rapidocr cannot be used because onnxruntime is not installed` on first boot | Only appears if the old image (before `onnxruntime` was added) is still cached. Force rebuild with `--no-cache`. |
