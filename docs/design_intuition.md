# Memory Schema Design Intuition

## Database Setup

### Prerequisites

In PostgreSQL, create a database named `poc_to_prod` and a user named `XYZ` with a secure password. Grant all privileges on the database to the user.

### Create Schema

```sql
CREATE SCHEMA IF NOT EXISTS memory;
```

### Enable Required Extensions

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
```

## Schema Design Rationale

### Initial Requirements

When designing a chat history table, the very basic columns needed are:

- **MESSAGE_ID** - Unique identifier for each message
- **SENDER** - Whether the message is from HUMAN or ASSISTANT
- **MESSAGE** - The actual message content
- **CREATED_AT** - Timestamp of when the message was created

### Why This Design Is Insufficient

#### Multi-User Isolation

A simplistic design with only the basic columns cannot prevent users from seeing each other's messages. We need **USER_ID** to isolate conversations per user.

#### Session Management

A single user may have multiple conversations with the assistant. We need **SESSION_ID** to manage conversations as distinct sessions, allowing users to:

- Maintain multiple active conversations
- View conversation history by session
- Isolate context between different conversations

### Refined Design

The relational design of our memory system includes:

- **user_id** - Link messages to specific users
- **session_id** - Organize messages into conversations
- **message_id** - Unique identifier for each message

## Final Schema Design

### Users Table

Stores information about each user in the system.

```sql
CREATE TABLE MEMORY.USERS (
    user_id UUID PRIMARY KEY DEFAULT UUID_GENERATE_V4(),
    name VARCHAR(255),
    email VARCHAR(255),
    password TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### Sessions Table

Organizes conversations by user. Each session represents a distinct conversation thread.

```sql
CREATE TABLE MEMORY.SESSIONS (
    session_id UUID PRIMARY KEY DEFAULT UUID_GENERATE_V4(),
    user_id UUID NOT NULL,
    session_name VARCHAR(60),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    terminated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### Chats Table

Stores the actual messages within each session.

```sql
CREATE TABLE MEMORY.CHATS (
    chat_id UUID PRIMARY KEY DEFAULT UUID_GENERATE_V4(),
    session_id UUID NOT NULL,
    sender TEXT NOT NULL CHECK (SENDER IN ('USER', 'ASSISTANT')),
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

## Benefits of This Design

This three-table design allows us to:

- Manage multiple users independently
- Support multiple sessions per user
- Isolate message history by session
- Track conversation metadata (session names, timestamps)
- Maintain audit trails (creation and termination times)
- Enable conversation resumption and history retrieval

---

## Implementation: From Notebook to Application

The `notebooks/explore_memory.ipynb` notebook was used to validate the schema and operations before wiring them into the application. The thought process followed these steps:

### Step 1 — Validate the schema works

Before building any service layer, raw `psycopg2` queries were used directly in the notebook to confirm the schema was correct. This surfaces issues early — wrong column names, missing constraints, type mismatches — without any application code in the way.

### Step 2 — Password security from the start

Passwords are hashed with `bcrypt` before storage. Plain-text passwords are never written to the database. The notebook uses `bcrypt.gensalt()` + `bcrypt.hashpw()` at insert time, and `bcrypt.checkpw()` at login time. This is the same pattern carried into `MemoryRepository.authenticate_user`.

### Step 3 — Prove the full conversation flow in isolation

The notebook runs the complete round-trip: create user → create session → add user message → call the LLM → add assistant message → fetch history. This confirmed the DB operations and the `ChatService` could be composed together before any UI existed.

### Step 4 — Extract into a reusable repository

Once the notebook queries were stable, they were moved into `src/memory/repository.py` as a `MemoryRepository` class. Each method opens and closes its own connection — this is intentional. Chainlit's async lifecycle means shared connections would require careful lifecycle management; per-method connections keep things simple and correct.

### Step 5 — Conversation history as LLM context

The notebook showed that fetching history per session is a simple `SELECT ... ORDER BY created_at ASC`. In the application, this history is injected into the LLM's system prompt as a labelled conversation block. The `ChatService._build_system_prompt` method handles this — if no history exists, the prompt is unchanged.

### Step 6 — Surface sessions in the UI

The Chainlit app renders sessions as `cl.Action` buttons. Switching a session updates `active_session_id` in `cl.user_session` and replays that session's history into the chat view. Creating a new session or ending the current one goes through `MemoryRepository` and refreshes the panel immediately.

### What this enables

- Each user sees only their own sessions and messages.
- Switching sessions gives the LLM the correct conversation history for that session, not a mix of all sessions.
- The LLM's responses are contextually aware of what was discussed earlier in the same session.
- Sessions can be terminated and preserved for later reference.