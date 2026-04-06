# Design Intuition

---

## Part 1 — Memory Schema

### Database Setup

In PostgreSQL, create a database named `poc_to_prod`. Extensions and schema are created by `sql/init.sql`.

```sql
CREATE SCHEMA IF NOT EXISTS poc2prod;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
```

### Schema Design Rationale

#### Initial Requirements

When designing a chat history table, the very basic columns needed are:

- **MESSAGE_ID** — unique identifier for each message
- **SENDER** — whether the message is from the user or the assistant
- **MESSAGE** — the actual message content
- **CREATED_AT** — timestamp of when the message was created

#### Why This Design Is Insufficient

A simplistic design with only the basic columns cannot prevent users from seeing each other's messages. We need **USER_ID** to isolate conversations per user.

A single user may also have multiple conversations. We need **SESSION_ID** to manage these as distinct threads, allowing users to maintain multiple active conversations and switch between them.

### Final Memory Schema

#### Users Table

```sql
CREATE TABLE poc2prod.users (
    user_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name         VARCHAR(255),
    email        VARCHAR(255) NOT NULL UNIQUE,
    password     TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

#### Sessions Table

```sql
CREATE TABLE poc2prod.sessions (
    session_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id      UUID NOT NULL REFERENCES poc2prod.users(user_id) ON DELETE CASCADE,
    session_name VARCHAR(60),
    is_active    BOOLEAN DEFAULT TRUE,
    created_at   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    terminated_at TIMESTAMPTZ
);
```

#### Chats Table

```sql
CREATE TABLE poc2prod.chats (
    chat_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES poc2prod.sessions(session_id) ON DELETE CASCADE,
    sender     TEXT NOT NULL CHECK (sender IN ('user', 'assistant')),
    message    TEXT NOT NULL,
    embeddings VECTOR,       -- for semantic history search
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

The `embeddings` column on chats uses `VECTOR` without a fixed dimension. This means the schema accepts any embedding size — the active provider's dimension is enforced at the application layer via `EmbeddingConfig.dimension`, not in DDL.

### Implementation: From Notebook to Application

The `notebooks/explore_memory.ipynb` notebook validated the schema and operations before wiring them into the application.

**Step 1 — Validate the schema works.** Raw `psycopg2` queries were used directly in the notebook. This surfaces wrong column names, missing constraints, and type mismatches without any application code in the way.

**Step 2 — Password security from the start.** Passwords are hashed with `bcrypt` before storage. Plain-text passwords are never written to the database.

**Step 3 — Prove the full conversation flow in isolation.** The notebook runs the complete round-trip: create user → create session → add user message → call the LLM → add assistant message → fetch history.

**Step 4 — Extract into a reusable repository.** Once the notebook queries were stable, they moved into `src/memory/repository.py` as a `MemoryRepository` class. Each method opens and closes its own connection — per-method connections keep things simple and avoid lifecycle issues.

**Step 5 — Conversation history as LLM context.** Fetching history per session is a simple `SELECT ... ORDER BY created_at ASC`. In the application, this history is injected into the LLM's system prompt as a labelled conversation block via `ChatService._build_system_prompt`.

---

## Part 2 — Document Ingestion and RAG

### Why RAG

The LLM's knowledge is frozen at its training cut-off and has no awareness of user-uploaded documents. RAG (Retrieval-Augmented Generation) bridges this: relevant excerpts from uploaded files are retrieved at query time and injected into the system prompt so the LLM can answer grounded in the actual document content.

### The Extraction Pipeline

Before text can be chunked or embedded, it must be extracted from the raw file. The extraction pipeline runs in three stages, all sharing an `ExtractionContext` object:

```text
ExtractionContext(file_path)
    → LayoutExtractor().extract(context)   # single Docling pass, discovers all elements
    → TextExtractor().extract(context)     # returns text + latex records from layout
```

`TextExtractor` is lightweight and dependency-free — it simply reads `record_type in ("text", "latex")` records from the already-computed layout. Tables and images are handled by separate extractors (`TableExtractor`, `ImageExtractor`) that can be layered in as needed.

Each record is an `ExtractedRecord` dataclass:

```python
@dataclass
class ExtractedRecord:
    record_type: str          # "text" | "table" | "image" | "url" | "latex"
    page: Optional[int]
    bbox: Optional[dict]      # Docling BOTTOMLEFT coordinates
    content: Any              # str for text/latex, dict for table/image
    raw: Optional[str] = None
```

**Key lesson from the notebook:** `ExtractedRecord` is a dataclass — access fields as attributes (`record.content`, `record.page`), never as dict keys (`record["content"]`).

### Chunking Strategy

Three chunking strategies are available, all sharing the `BaseChunker` interface:

| Strategy | Class | How it works | When to use |
| --- | --- | --- | --- |
| Hierarchical | `HierarchicalChunker` | Parent chunks (2000 chars) → child chunks (400 chars) | Default for RAG |
| Lexical semantic | `TextTilingChunker` | Bag-of-words cosine similarity, valley detection | No model, fast |
| Embedding semantic | `EmbeddingSemanticChunker` | Embedding cosine similarity between sentences | Best quality, slower |

For this application, **hierarchical chunking** is the chosen strategy.

#### Why Hierarchical Chunking

Dense retrieval works best on small, precise chunks (low noise, high signal). But when a child chunk is returned, its narrow window often lacks the full context the LLM needs to construct a good answer. Hierarchical chunking solves this with a two-level split:

- **Child chunks (400 chars)** — indexed in the vector store for retrieval. Small, focused, high precision.
- **Parent chunks (2000 chars)** — not indexed, but fetched at retrieval time using the child's FK. Passed to the LLM as the actual context. Rich, complete, lower noise for generation.

This is called "small-to-large" or "parent-document" retrieval.

```python
parent_docs, child_docs = HierarchicalChunker().chunk_with_parents(text, metadata=meta)
# Each child doc carries parent_id (int index) in its metadata
# parent_id maps to a row in poc2prod.parenthierarchy via UUID
```

#### The `parent_id` integer → UUID mapping problem

`HierarchicalChunker` assigns each parent chunk an integer index (`0, 1, 2, ...`). The database uses UUIDs. The solution is to build a mapping dict during parent insertion:

```python
parent_index_to_uuid: dict[int, str] = {}
for idx, parent_doc in enumerate(parent_docs):
    row = await conn.fetchrow("INSERT INTO poc2prod.parenthierarchy ... RETURNING id;", ...)
    parent_index_to_uuid[idx] = str(row["id"])
```

Then during child insertion, look up `meta["parent_id"]` (the int) to get the actual UUID FK.

### Embedding Providers

Three providers share the `BaseEmbedder` interface (`embed(texts) → list[list[float]]`):

| Provider | Class | Dimension | Notes |
| --- | --- | --- | --- |
| Local (sentence-transformers) | `LocalEmbedder` | 384 | Fully offline, smallest |
| Ollama | `OllamaEmbedder` | 1024 | Local server, no API key |
| OpenAI | `OpenAIEmbedder` | 1536 | Best quality, costs money |

The active provider and its dimension are read from `configs/config.yaml` at startup via `ConfigManager.embedding_config`. Switching providers requires only a one-line change in the config — no code changes.

**Key lesson:** The embedding dimension is a runtime property of the provider, not a DDL constraint. Using `VECTOR` (no dimension) in the schema lets any provider write to the same column without a migration. The IVFFlat index (which does require a fixed dimension) is built post-ingestion, not in `init.sql`.

### Database Schema for RAG

Two tables are added to `poc2prod`:

#### parenthierarchy

Stores large parent chunks produced by `HierarchicalChunker`. Not searched directly — fetched by UUID after a child chunk is matched.

```sql
CREATE TABLE poc2prod.parenthierarchy (
    id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_chunk_content TEXT,
    filename             VARCHAR(500) NOT NULL,
    metadata             JSONB DEFAULT '{}'::jsonb,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);
```

#### ingestions

Stores small child chunks with their embeddings. This is the table searched at query time.

```sql
CREATE TABLE poc2prod.ingestions (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_id        UUID REFERENCES poc2prod.parenthierarchy(id) ON DELETE SET NULL,
    user_id          UUID REFERENCES poc2prod.users(user_id) ON DELETE SET NULL,
    session_id       UUID REFERENCES poc2prod.sessions(session_id) ON DELETE SET NULL,
    filename         VARCHAR(500) NOT NULL,
    file_description TEXT,
    type             VARCHAR(50) NOT NULL CHECK (type IN ('pdf', 'doc')),
    chunk_content    TEXT NOT NULL,
    embeddings       VECTOR,      -- dimension matches active provider
    metadata         JSONB DEFAULT '{}'::jsonb,
    version          FLOAT DEFAULT 1.0,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);
```

**Design decisions:**

- `parent_id` uses `ON DELETE SET NULL` — losing a parent chunk is non-fatal; the child is still searchable.
- `user_id` and `session_id` are nullable FKs so retrieval can be scoped to a session without requiring joins.
- `embeddings` is dimensionless `VECTOR` — dimension enforcement lives in the application layer.

### Vector Search and Retrieval

pgvector's `<=>` operator computes cosine distance. `1 - distance` gives cosine similarity (higher = more similar):

```sql
SELECT
    id::text AS child_id,
    parent_id::text,
    chunk_content,
    1 - (embeddings <=> $1::vector) AS similarity
FROM poc2prod.ingestions
WHERE session_id = $2
ORDER BY embeddings <=> $1::vector
LIMIT $3;
```

The vector is passed as a string literal `"[v1,v2,...]"` with a `::vector` cast — asyncpg does not have a native pgvector codec, so string serialisation is the correct approach.

**Key lesson on `search_path`:** The `vector` type and `ivfflat` access method live in the `public` schema. asyncpg connections default to only the user's own schema. The search path must be set at connect time:

```python
await asyncpg.connect(
    **DB_CONFIG,
    server_settings={"search_path": "poc2prod,public"},
)
```

Without this, `type "vector" does not exist` and `operator class "vector_cosine_ops" does not exist` errors occur even when the extension is installed.

### The IVFFlat Index

An IVFFlat index makes cosine search fast at scale, but it requires:

1. A consistent vector dimension (which `VECTOR` without dimension doesn't enforce in DDL)
2. Enough rows already loaded (`~lists * 39` minimum, default `lists = 100` means ~3900 rows)

For these reasons the index is **not** in `init.sql`. It is created post-ingestion, after a full document load, via:

```sql
CREATE INDEX IF NOT EXISTS idx_ingestions_embeddings_cosine
ON poc2prod.ingestions
USING ivfflat (embeddings vector_cosine_ops)
WITH (lists = 100);
```

### Ingestion: From Notebook to Application

The `notebooks/explore_ingestion.ipynb` notebook validated every stage before wiring into the application.

**Stage 1 — Extraction.** `LayoutExtractor` runs a single Docling pass over the PDF, producing a `LayoutResult`. `TextExtractor` then filters to `("text", "latex")` records. Using the dataclass attributes directly (`record.content`, `record.page`) is required — `record["content"]` raises `TypeError`.

**Stage 2 — Chunking.** `HierarchicalChunker.chunk_with_parents()` returns `(parent_docs, child_docs)`. Each child carries `parent_id` (int) and `parent_text` in its metadata. When iterating `text_records`, non-text types and empty strings must be filtered before calling `chunk_with_parents`.

**Stage 3 — Embedding.** The active embedder is instantiated from `ConfigManager.embedding_config`. The dimension from config must match what the model actually outputs — mismatches cause silent corruption at query time.

**Stage 4 — Insertion.** Parents are inserted first; their returned UUIDs are mapped from the chunker's integer indices. Children are inserted with the resolved UUID FK. Vectors are serialised as `"[v1,v2,...]"` strings.

**Stage 5 — Retrieval.** The query is embedded with the same embedder used at ingestion time. The top-K child chunks are fetched via cosine similarity. Their parent UUIDs are collected and fetched in a single `WHERE id = ANY($1::uuid[])` query. The parent text (not the child text) is assembled into the RAG context block passed to the LLM.

**What this enables:**

- The LLM answers questions grounded in the actual uploaded documents.
- Retrieval is session-scoped — users only retrieve from their own uploaded files.
- Switching embedding providers requires one config line change; no schema migration needed.
- The parent-document pattern gives the LLM a richer context window than raw child chunks alone.
