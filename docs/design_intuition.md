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

**Step 5 — Conversation history as LLM context.** Fetching history per session is a simple `SELECT ... ORDER BY created_at ASC`. In the application, this history is injected into the LLM's system prompt via `ChatService._build_system_prompt`. See Part 3 for how the history is split into short-term and long-term memory.

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

---

## Part 3 — Short-Term and Long-Term Memory

### The Problem with Unbounded History

The initial design injected the entire session history into the system prompt on every turn. This has two failure modes:

1. **Token limit exhaustion** — long sessions silently degrade response quality and eventually exceed the LLM's context window.
2. **Noise over signal** — old, unrelated exchanges dilute the context, making the LLM less focused on what is currently relevant.

### The Two-Layer Memory Architecture

Memory is now split into two complementary layers:

```text
Every chat turn
  │
  ├─ Short-term memory  — last N messages (chronological recency)
  │
  └─ Long-term memory   — top-K semantically similar past messages
                          (relevance, not recency)
```

Both layers feed into the system prompt, with duplicates removed. The LLM always sees the most recent context **and** the most relevant historical context, without redundancy or runaway growth.

### Short-Term Memory

Short-term memory is the last `short_term_limit` messages from the session, fetched chronologically. It is the direct successor to the original unbounded history — same idea, bounded by config.

```yaml
# configs/config.yaml
chat:
  short_term_limit: 10
```

The query fetches `short_term_limit + 1` rows. The extra row serves a dual purpose: it provides the just-added user message (which is then sliced off), and its presence signals that the session has crossed the threshold needed to activate long-term memory — no separate `COUNT` query required.

### Long-Term Memory

Long-term memory is a cosine similarity search over **all past messages in the session that have stored embeddings**. It retrieves up to 10 messages whose semantic content is most similar to the current user query, regardless of when they occurred.

#### When it activates

Long-term memory is only queried when the session history has **already crossed** `short_term_limit`. The guard condition uses the same `+1` fetch described above: if `len(raw_history) == short_term_limit + 1`, there is history older than what short-term covers and the semantic search is meaningful. Otherwise it is skipped entirely — no vector search, no embedder call.

```text
Session has ≤ short_term_limit messages → long-term skipped
Session has > short_term_limit messages → long-term search runs
```

#### Similarity threshold

Not every semantic match is worth including. A result that is only marginally related would add noise. A minimum cosine similarity threshold filters the raw results:

```yaml
# configs/config.yaml
chat:
  long_term_similarity_threshold: 0.50
```

Results below this value are dropped before the long-term list is finalised.

#### Deduplication

After the threshold filter, any message already present in short-term memory is removed from the long-term list by `chat_id`. This prevents the same message from appearing twice in the system prompt.

### How Embeddings Are Stored

Every user message and assistant reply is now embedded and stored in the `chats.embeddings` column at write time:

- **User message** — embedded with `embedder.embed_query()` (uses the query-task prefix for asymmetric models). The same vector is reused for RAG retrieval and long-term memory search — no double embedding.
- **Assistant message** — embedded with `embedder.embed_one()` (uses the document-task prefix).
- **Blocked/error replies** — stored without an embedding (`embeddings IS NULL`). These rows are invisible to the semantic search.

The `chats.embeddings` column existed in the schema from the `feature/rag` branch but was always NULL. It is now actively written.

### System Prompt Structure

The final system prompt is assembled in this order (closest to the current question last):

```text
[Base system prompt]
[Relevant Document Excerpts]   ← RAG context (if documents uploaded)
[Relevant Past Exchanges]      ← Long-term memory (if active and non-empty)
[Recent Conversation]          ← Short-term memory
```

### Observability

Every turn logs the memory pipeline at INFO level:

```text
[memory] session=... | short-term fetched: 10 (limit=10)
[memory] session=... | long-term raw results: 7
[memory] session=... | long-term after threshold (>= 0.5): 4
[memory] session=... | long-term after dedup: 3 (1 duplicate(s) removed)
```

Or when the guard fires:

```text
[memory] session=... | short-term fetched: 6 (limit=10)
[memory] session=... | long-term skipped (session has not yet crossed short_term_limit=10)
```

### Key Design Decisions

| Decision | Rationale |
| --- | --- |
| `short_term_limit + 1` fetch instead of a `COUNT` query | One DB round-trip detects the threshold and retrieves the data simultaneously |
| Threshold in `config.yaml`, not hardcoded | Tunable without code changes — raise it for precision, lower it for broader recall |
| Guard on session length | Prevents wasted vector search on short sessions where short-term already covers everything |
| Reuse `query_vec` across RAG and long-term search | Embed once per turn regardless of how many retrieval systems consume the vector |
| Blocked replies stored without embeddings | Toxic/blocked exchanges should not influence future semantic retrieval |

---

## Part 4 — LangGraph Orchestration, Reranking, and Access Control

### Why an Orchestrator Layer

The earlier RAG pipeline was a fixed linear sequence: embed → retrieve → generate. This works for simple factual queries but breaks down for:

- **Ambiguous or multi-part questions** — a single retrieval pass over a poorly-phrased query returns noisy chunks
- **Complex research questions** — multiple sub-topics need independent retrieval threads
- **Quality control** — there is no feedback loop to catch weak or hallucinated answers

The orchestrator layer adds intelligence before and after retrieval, without coupling that logic to the HTTP handler.

### Fast vs. Deep Mode

Two modes are exposed via the `mode` field in `SendMessageRequest`:

```
mode: "fast"   (default) — optimised for latency
mode: "deep"              — optimised for answer quality
```

#### Fast Mode Graph

```
resolve_memory → retrieve → rerank_and_build_context → generate
```

No extra LLM calls. Memory is resolved once and injected into the generation system prompt. Retrieval and reranking happen in sequence. This path adds no more than one extra LLM call beyond baseline.

#### Deep Mode Graph

```
resolve_memory
  → analyze_query          (LLM: intent, complexity, clarity)
      ├─ unclear intent  → query_clarification  (interrupt, await user)
      └─ clear intent
            ├─ low complexity  → (same as fast from here)
            └─ high complexity
                  ├─ single topic  → query_rewrite → retrieve
                  └─ multi-topic   → query_decompose → [retrieve_sub_query × N fan-out]
  → rerank_and_build_context
  → generate
  → validate_response      (LLM-as-judge)
      ├─ pass  → finalize
      └─ fail  → correction → generate  (loop, max 3 iterations; returns best_response on cap)
```

#### Key design choices

**Low complexity in deep mode is treated as fast mode.** An explicit `analyze_query` LLM call already happened, so the cost is sunk. The value of deep mode's query rewrite and validation only justifies itself for complex questions — simple ones get routed directly to retrieval.

**`route_complexity` is a pass-through node, not a function.** LangGraph requires a named node to attach conditional edges from. `lambda s: s` is the node body; the routing logic lives in the conditional edge function attached to it.

**`interrupt()` for HITL.** When intent is unclear, the graph calls `interrupt(clarification_question)`. This pauses the graph and serialises state to `MemorySaver`. The SSE endpoint detects the interrupt via `graph.get_state(thread_id).next`, emits a `clarification` event to the frontend, and stores `session_id → thread_id` in `app.state.pending_clarifications`. The user's reply arrives on the next POST; `Command(resume=user_reply)` resumes the paused graph from the exact point it stopped.

**`raw_chunks: Annotated[list[dict], operator.add]`** — the `operator.add` reducer lets all parallel `retrieve_sub_query` nodes (spawned via `Send`) accumulate their results into a single list without conflict.

**Validation loop cap.** `iteration_count` is incremented on each `correction → generate` cycle. When `iteration_count >= 3`, the loop exits and returns `best_response` (the highest-quality answer seen so far) rather than the potentially degraded final answer.

**Memory injection at generation only.** Short-term and long-term memory are resolved once (`resolve_memory` node) and injected into the system prompt only at the `generate` node — not into the retrieval query. Using conversation context to bias retrieval tends to over-weight recent exchanges and under-weight document relevance.

### Reranker

The retrieval step returns the top-K chunks by cosine similarity. Cosine similarity alone is a weak signal — it scores embedding space proximity, not semantic relevance to the specific query phrasing.

A cross-encoder reranker rescores `(query, passage)` pairs jointly: the query and passage are passed together as a single input, so the model can attend across both. This is slower than bi-encoder similarity but produces much more accurate relevance scores.

```
BaseReranker
  └── CrossEncoderReranker          (sentence-transformers CrossEncoder)
```

`CrossEncoderReranker.rerank(query, chunks, top_k)`:
1. Builds `(query, chunk_content)` pairs
2. Calls `model.predict(pairs)` — one forward pass per pair
3. Sorts descending by score, adds `rerank_score` key to each dict
4. Returns top-k chunks

Configuration in `configs/config.yaml`:

```yaml
reranker:
  enabled: true
  model: "BAAI/bge-reranker-base"   # bge-reranker-base | bge-reranker-large | bge-reranker-v2-m3
  top_k: 5
  device: "cpu"                      # or "cuda"
```

Switching the model requires only a config change — no code changes.

### SSE Streaming and Per-Node Status

The `stream_message` endpoint uses `graph.astream(stream_mode="updates", subgraphs=True)`. Each yielded item is a `(namespace_tuple, {node_name: state_delta})` tuple.

- **Outer graph nodes** have `namespace = ()`
- **Inner subgraph nodes** (FastOrchestrator, DeepOrchestrator) have a non-empty namespace

`_DEEP_NODE_STATUS` maps node names to human-readable status strings. The endpoint:
1. Emits a `status` SSE event when the mapped string changes (deduplicated to avoid repeating "Searching documents…" three times for decomposed queries)
2. Does not emit a status event for nodes mapped to `None` (`query_clarification`, `finalize`)
3. After streaming completes, calls `get_graph_state()` to read the final state and emit the accumulated response as `token` events

In **fast mode**, status events are suppressed entirely — answers arrive fast enough that progress indicators would flash and disappear before the user reads them.

### Database Changes for Orchestration

The `chats` table gains an `orchestrator_metadata JSONB` column (default `'{}'::jsonb`) that stores per-message orchestration data such as `mode`, `query_intent`, `query_complexity`, and `iteration_count`. An index on `orchestrator_metadata->>'mode'` supports future analytics queries.

### User Access Control

#### The Problem

Without access control, anyone who can reach the signup endpoint can immediately begin using the application. For a private deployment this is unacceptable — it exposes the LLM and document store to arbitrary users.

#### The Design

New signups are stored with `status = 'pending'` in the `users` table. The admin approves or rejects requests by running a direct SQL update. This requires no admin UI and leaves a clear audit trail in the database.

```sql
-- users table gains:
status VARCHAR(20) NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'approved', 'rejected'))
```

**Signup** — always returns HTTP 201 with a `{ message, status: "pending" }` response. No token is issued. The frontend shows a "request submitted, awaiting approval" screen.

**Sign-in** — `authenticate_user()` checks `status` after verifying the password:
- `'pending'` → `UserNotApprovedError("pending")` → HTTP 403 "Your account is awaiting admin approval."
- `'rejected'` → `UserNotApprovedError("rejected")` → HTTP 403 "Your account access has been declined."
- `'approved'` → proceeds normally, issues JWT

The password is checked before the status check so that the error message for a wrong password (`401 Invalid email or password`) leaks no information about whether the email is registered.

**Admin workflow:**

```sql
-- See all pending requests
SELECT user_id, name, email, created_at FROM poc2prod.users WHERE status = 'pending' ORDER BY created_at;

-- Approve
UPDATE poc2prod.users SET status = 'approved' WHERE email = 'user@example.com';

-- Reject
UPDATE poc2prod.users SET status = 'rejected' WHERE email = 'user@example.com';
```

---

## Part 5 — Agentic RAG Design

### Why add agents when workflows already exist

The deterministic workflow layer solves a large class of RAG problems very well:

- `fast` is cheap, predictable, and good for straightforward document QA
- `deep` is better for ambiguity, decomposition, and iterative correction

But fixed graphs become awkward when the question needs adaptive evidence gathering:

- compare uploaded documents with current external knowledge
- combine document evidence with exact calculations
- decide dynamically whether one search pass is enough or whether a second tool call is needed

This is where agentic RAG becomes useful. The model is allowed to plan its own
tool usage, but only within carefully chosen boundaries.

### The key boundary: adaptive reasoning, deterministic infrastructure

The central design decision is:

> make planning agentic, keep infrastructure deterministic

That means the following still remain fixed backend concerns:

- short-term and long-term memory resolution
- embedding creation
- vector retrieval internals
- reranking internals
- parent-context fetching
- persistence of chat records and metadata

The agent is responsible for:

- deciding whether tools are needed
- choosing which high-level tool to call
- deciding when it has enough evidence
- synthesizing the final answer

This separation prevents the system from degenerating into "LLM controls every backend primitive", which is powerful in theory but fragile and hard to debug in practice.

### Why low-level retrieval primitives are not agent tools

It is tempting to expose backend primitives such as:

- `embed_query`
- `search_pgvector`
- `rerank_chunks`
- `fetch_parent_contexts`

This project does **not** do that.

Those primitives are implementation details. They are meaningful to engineers,
but not the right abstraction level for the model.

Instead, the agent sees user-meaningful tools:

- `search_documents`
- `get_uploaded_documents`
- `summarize_document`
- `extract_paper_metadata`
- `web_search`
- `fetch_webpage`
- `calculate`

Each of these tools encapsulates its own lower-level pipeline. This produces:

- fewer tool calls
- better reliability
- lower latency
- cleaner observability
- easier future refactors of retrieval internals

### Single-agent design

The first agent mode is `single_rag_agent`.

Graph:

```text
resolve_memory → run_agent → END
```

The agent gets all high-level tools in one toolbox:

- document tools
- web tools
- calculation tool

This mode exists because it gives the user agentic flexibility with minimal
coordination overhead. It is the natural first step before true multi-agent
delegation.

### Why a supervisor model was added later

Once a single agent is working, the next useful improvement is not "more tools".
It is better separation of responsibility.

Some tasks are naturally multi-specialist:

- "Compare what this uploaded paper claims with recent public information"
- "Pull the metrics from the paper and compute the relative improvement"
- "Summarize the uploaded report and check whether recent external sources agree"

These are good fits for a supervisor-plus-workers design.

### The worker split: role-based, not subsystem-based

The worker agents in this project are:

- **Document Research Worker**
- **Web Research Worker**
- **Computation Worker**

This is intentionally a role-based split.

Rejected alternatives included workers such as:

- retriever worker
- reranker worker
- embedding worker
- memory worker

Those roles mirror backend subsystems, not meaningful research behaviors. They
would create more coordination overhead without creating better specialist
reasoning.

### Worker responsibilities

#### Document Research Worker

Allowed tools:

- `get_uploaded_documents`
- `search_documents`
- `summarize_document`
- `extract_paper_metadata`

Its job is to stay grounded in uploaded session documents and avoid web assumptions.

#### Web Research Worker

Allowed tools:

- `web_search`
- `fetch_webpage`

Its job is to gather current or external information and return concise findings.

#### Computation Worker

Allowed tools:

- `calculate`

Its job is to solve only the numerical part of the problem with exact outputs.

### Delegation as worker-facing tools

The supervisor does not directly own all worker tools. Instead, it receives
three delegation tools:

- `ask_document_worker(task)`
- `ask_web_worker(task)`
- `ask_computation_worker(task)`

Each delegation tool runs a specialized worker agent behind the scenes.

This design is important for three reasons:

1. specialization is enforced by construction
2. worker behavior is easier to inspect and log
3. the supervisor remains a planner/synthesizer rather than becoming another
   single-agent "everything tool" wrapper

### Supervisor design

The supervisor's role is:

1. understand the user request
2. decide whether the task needs document evidence, web evidence, math, or a combination
3. delegate only when necessary
4. synthesize worker outputs into the final answer

The supervisor is **not** intended to perform low-level retrieval itself.

### Validation strategy for agents

The agentic paths currently skip the deep-mode validation loop.

This is a deliberate tradeoff:

- agentic reasoning already adds latency
- adding LLM-as-judge on top makes iteration slower and harder to inspect
- first-pass agent development is easier when there is only one active reasoning layer

This does **not** mean validation is useless for agents forever. It means the
first implementation optimizes for observability and controllable complexity.

### Unified API and UI model

The earlier API exposed a single `mode` field. That was sufficient for
`fast` vs `deep`, but it became too ambiguous once agents were introduced.

The API now uses:

```json
{
  "category": "workflow" | "agent",
  "variant":  "fast" | "deep" | "single_rag_agent" | "supervisor_orchestration_agent"
}
```

This matches the frontend model:

- **Workflows**
  - Fast
  - Deep
- **Agents**
  - Single RAG Agent
  - Supervisor Agent

This naming keeps the conceptual difference explicit:

- workflows = fixed graph logic
- agents = adaptive tool-using reasoning
