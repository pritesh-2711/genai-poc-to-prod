-- Database: poc_to_prod

-- DROP DATABASE IF EXISTS poc_to_prod;

-- CREATE DATABASE poc_to_prod
--     WITH
--     OWNER = postgres
--     ENCODING = 'UTF8'
--     LC_COLLATE = 'en_US.UTF-8'
--     LC_CTYPE = 'en_US.UTF-8'
--     LOCALE_PROVIDER = 'libc'
--     TABLESPACE = pg_default
--     CONNECTION LIMIT = -1
--     IS_TEMPLATE = False;

-- GRANT TEMPORARY, CONNECT ON DATABASE poc_to_prod TO PUBLIC;

-- GRANT ALL ON DATABASE poc_to_prod TO postgres;

-- GRANT TEMPORARY ON DATABASE poc_to_prod TO "pritesh-jha";


CREATE SCHEMA IF NOT EXISTS poc2prod;
SET search_path TO poc2prod, public;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================================
-- TABLE: users
-- ============================================================================

CREATE TABLE poc2prod.users (
    user_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name         VARCHAR(255),
    email        VARCHAR(255) NOT NULL UNIQUE,
    password     TEXT NOT NULL,
    status       VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TABLE: sessions
-- ============================================================================

CREATE TABLE poc2prod.sessions (
    session_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id      UUID NOT NULL REFERENCES poc2prod.users(user_id) ON DELETE CASCADE,
    session_name VARCHAR(60),
    is_active    BOOLEAN DEFAULT TRUE,
    created_at   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    terminated_at TIMESTAMPTZ
);

-- ============================================================================
-- TABLE: chats
-- Stores conversation turns. embeddings holds the query/response vector
-- for semantic history search and RAG context retrieval.
-- ============================================================================

CREATE TABLE poc2prod.chats (
    chat_id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id             UUID NOT NULL REFERENCES poc2prod.sessions(session_id) ON DELETE CASCADE,
    sender                 TEXT NOT NULL CHECK (sender IN ('user', 'assistant')),
    message                TEXT NOT NULL,
    embeddings             VECTOR,       -- dimension matches active embedder provider (see config.yaml)
    -- Orchestrator metadata: mode, query_complexity, iteration count, validation result, etc.
    orchestrator_metadata  JSONB DEFAULT '{}'::jsonb,
    created_at             TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TABLE: parenthierarchy
-- Stores parent (large) chunks produced by HierarchicalChunker.
-- Child chunks in `ingestions` reference rows here via parent_id.
-- ============================================================================

CREATE TABLE IF NOT EXISTS poc2prod.parenthierarchy (
    -- Primary Key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Parent chunk text (denormalized for fast retrieval)
    parent_chunk_content TEXT,

    -- Document Metadata
    filename VARCHAR(500) NOT NULL,

    -- Additional Metadata (source, page range, etc.)
    metadata JSONB DEFAULT '{}'::jsonb,

    content_type VARCHAR(20) NOT NULL DEFAULT 'text' 
        CHECK (content_type IN ('text', 'table', 'image')),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- TABLE: ingestions
-- Stores child (small) chunks that are indexed for vector search.
-- Each row optionally links back to its parent chunk via parent_id.
-- When chunking method is NOT hierarchical, parent_id is NULL.
-- ============================================================================

CREATE TABLE IF NOT EXISTS poc2prod.ingestions (
    -- Primary Key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Hierarchy link (NULL for non-hierarchical chunking)
    parent_id UUID REFERENCES poc2prod.parenthierarchy(id) ON DELETE SET NULL,

    -- Ownership
    user_id    UUID REFERENCES poc2prod.users(user_id) ON DELETE SET NULL,
    session_id UUID REFERENCES poc2prod.sessions(session_id) ON DELETE SET NULL,

    -- Document Metadata
    filename         VARCHAR(500) NOT NULL,
    file_description TEXT,

    -- File Type Classification (Research papers are usually pdf, or docx)
    type VARCHAR(50) NOT NULL
        CHECK (type IN ('pdf', 'doc')),

    content_type VARCHAR(20) NOT NULL DEFAULT 'text'
        CHECK (content_type IN ('text', 'table', 'image')),

    -- Chunk Content (child chunk text)
    chunk_content TEXT NOT NULL,

    -- Embeddings — dimension matches active embedder provider (see config.yaml)
    embeddings VECTOR,

    -- Additional Metadata (page, bbox, source, etc.)
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Versioning
    version FLOAT DEFAULT 1.0,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX idx_sessions_user_id    ON poc2prod.sessions(user_id);
CREATE INDEX idx_chats_session_id    ON poc2prod.chats(session_id);

-- ingestions
CREATE INDEX idx_ingestions_parent_id   ON poc2prod.ingestions(parent_id);
CREATE INDEX idx_ingestions_user_id     ON poc2prod.ingestions(user_id);
CREATE INDEX idx_ingestions_session_id  ON poc2prod.ingestions(session_id);
CREATE INDEX idx_ingestions_filename    ON poc2prod.ingestions(filename);
CREATE INDEX idx_ingestions_metadata    ON poc2prod.ingestions USING GIN(metadata);

-- NOTE: IVFFlat / HNSW vector index on ingestions.embeddings is NOT created here.
-- It requires a fixed-dimension column and should be built after data is loaded:
--
--   CREATE INDEX idx_ingestions_embeddings_cosine ON poc2prod.ingestions
--       USING ivfflat (embeddings vector_cosine_ops) WITH (lists = 100);
--
-- Run src/databases/create_vector_index.py (or the notebook helper) which reads
-- the active provider dimension from config.yaml before creating the index.

-- parenthierarchy
CREATE INDEX idx_parenthierarchy_filename ON poc2prod.parenthierarchy(filename);
CREATE INDEX idx_parenthierarchy_metadata ON poc2prod.parenthierarchy USING GIN(metadata);

-- Index for filtering chunks by type during retrieval
CREATE INDEX IF NOT EXISTS idx_ingestions_content_type
    ON poc2prod.ingestions(content_type);

-- ============================================================================
-- TRIGGER FUNCTION: auto-update updated_at
-- ============================================================================

CREATE OR REPLACE FUNCTION poc2prod.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

CREATE TRIGGER update_ingestions_updated_at
    BEFORE UPDATE ON poc2prod.ingestions
    FOR EACH ROW
    EXECUTE FUNCTION poc2prod.update_updated_at_column();

CREATE TRIGGER update_parenthierarchy_updated_at
    BEFORE UPDATE ON poc2prod.parenthierarchy
    FOR EACH ROW
    EXECUTE FUNCTION poc2prod.update_updated_at_column();
