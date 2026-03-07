CREATE SCHEMA IF NOT EXISTS memory;
SET search_path TO memory, public;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

CREATE TABLE memory.users (
    user_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name         VARCHAR(255),
    email        VARCHAR(255) NOT NULL UNIQUE,
    password     TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE memory.sessions (
    session_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id      UUID NOT NULL REFERENCES memory.users(user_id) ON DELETE CASCADE,
    session_name VARCHAR(60),
    is_active    BOOLEAN DEFAULT TRUE,
    created_at   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    terminated_at TIMESTAMPTZ
);

CREATE TABLE memory.chats (
    chat_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES memory.sessions(session_id) ON DELETE CASCADE,
    sender     TEXT NOT NULL CHECK (sender IN ('user', 'assistant')),
    message    TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_user_id ON memory.sessions(user_id);
CREATE INDEX idx_chats_session_id ON memory.chats(session_id);