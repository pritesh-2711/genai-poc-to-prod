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
    sender UUID NOT NULL,
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
