-- Database: poc_to_prod

-- DROP DATABASE IF EXISTS poc_to_prod;

CREATE DATABASE poc_to_prod
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    LOCALE_PROVIDER = 'libc'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1
    IS_TEMPLATE = False;

GRANT TEMPORARY, CONNECT ON DATABASE poc_to_prod TO PUBLIC;

GRANT ALL ON DATABASE poc_to_prod TO postgres;

GRANT TEMPORARY ON DATABASE poc_to_prod TO "pritesh-jha";

-----------------------------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS application.vectordatatable (
    -- Primary Key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Ownership (for multi-tenancy)
    user_id UUID,
    session_id UUID,
    
    -- Document Metadata
    filename VARCHAR(500) NOT NULL,
    file_description TEXT,
    author VARCHAR(255),
    
    -- File Type Classification
    type VARCHAR(50) DEFAULT 'pdf' 
        CHECK (type IN ('pdf', 'excel', 'doc', 'txt', 'database', 'url')),
    
    -- Chunk Content
    chunk_content TEXT NOT NULL,
    
    -- Embeddings
    embeddings VECTOR(1536), -- 
    
    -- Additional Metadata (flexible JSON)
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Versioning
    version FLOAT DEFAULT 1.0,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- TABLE: hierarchytable
-- Purpose: Store parent-child relationships between chunks
-- ============================================================================

CREATE TABLE IF NOT EXISTS application.hierarchytable (
    -- Primary Key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Parent Reference
    parent_id UUID,
    
    -- Parent Chunk Content (denormalized for performance)
    parent_chunk_content TEXT,
    
    -- Document Metadata
    filename VARCHAR(500) NOT NULL,
    chapter_name VARCHAR(500),
    
    -- Additional Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Vectordatatable indexes

-- Ownership indexes
CREATE INDEX IF NOT EXISTS idx_vectordata_user_id ON application.vectordatatable(user_id);
CREATE INDEX IF NOT EXISTS idx_vectordata_session_id ON application.vectordatatable(session_id);

-- Document classification
CREATE INDEX IF NOT EXISTS idx_vectordata_type ON application.vectordatatable(type);
CREATE INDEX IF NOT EXISTS idx_vectordata_doctype ON application.vectordatatable(doctype);
CREATE INDEX IF NOT EXISTS idx_vectordata_filename ON application.vectordatatable(filename);

-- Language filtering (NEW)
CREATE INDEX IF NOT EXISTS idx_vectordata_language ON application.vectordatatable(content_language);

-- Chapter/theme search
CREATE INDEX IF NOT EXISTS idx_vectordata_chapter ON application.vectordatatable(chapter_name);
CREATE INDEX IF NOT EXISTS idx_vectordata_theme ON application.vectordatatable(theme);

-- Keywords search (GIN for array operations)
CREATE INDEX IF NOT EXISTS idx_vectordata_keywords ON application.vectordatatable USING GIN(keywords);

-- Metadata search (GIN for JSONB)
CREATE INDEX IF NOT EXISTS idx_vectordata_metadata ON application.vectordatatable USING GIN(metadata);

-- Vector similarity search (IVFFlat for approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_vectordata_embeddings_cosine ON application.vectordatatable 
USING ivfflat (embeddings vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_vectordata_embeddings_l2 ON application.vectordatatable 
USING ivfflat (embeddings vector_l2_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_vectordata_theme_vector ON application.vectordatatable 
USING ivfflat (theme_vector vector_cosine_ops) WITH (lists = 100);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_vectordata_doctype_keywords ON application.vectordatatable(doctype) 
WHERE keywords IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_vectordata_user_doctype ON application.vectordatatable(user_id, doctype);

-- Hierarchytable indexes
CREATE INDEX IF NOT EXISTS idx_hierarchy_parent_id ON application.hierarchytable(parent_id);
CREATE INDEX IF NOT EXISTS idx_hierarchy_filename ON application.hierarchytable(filename);
CREATE INDEX IF NOT EXISTS idx_hierarchy_chapter ON application.hierarchytable(chapter_name);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION application.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_vectordata_updated_at
    BEFORE UPDATE ON application.vectordatatable
    FOR EACH ROW
    EXECUTE FUNCTION application.update_updated_at_column();

CREATE TRIGGER update_hierarchy_updated_at
    BEFORE UPDATE ON application.hierarchytable
    FOR EACH ROW
    EXECUTE FUNCTION application.update_updated_at_column();

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Search by keywords (exact match)
CREATE OR REPLACE FUNCTION application.search_by_keywords(
    p_keywords TEXT[],
    p_doctype VARCHAR(50) DEFAULT NULL,
    p_language VARCHAR(50) DEFAULT NULL,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    chunk_content TEXT,
    filename VARCHAR(500),
    chapter_name VARCHAR(500),
    similarity_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        v.id,
        v.chunk_content,
        v.filename,
        v.chapter_name,
        1.0 as similarity_score -- Exact keyword match
    FROM application.vectordatatable v
    WHERE v.keywords && p_keywords -- Array overlap operator
    AND (p_doctype IS NULL OR v.doctype = p_doctype)
    AND (p_language IS NULL OR v.content_language = p_language)
    ORDER BY v.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Search by theme (fuzzy match using trigram similarity)
CREATE OR REPLACE FUNCTION application.search_by_theme(
    p_theme VARCHAR(500),
    p_similarity_threshold FLOAT DEFAULT 0.3,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    chunk_content TEXT,
    filename VARCHAR(500),
    theme VARCHAR(500),
    similarity_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        v.id,
        v.chunk_content,
        v.filename,
        v.theme,
        similarity(v.theme, p_theme) as similarity_score
    FROM application.vectordatatable v
    WHERE similarity(v.theme, p_theme) > p_similarity_threshold
    ORDER BY similarity_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Hybrid search (combine vector + keywords + theme)
CREATE OR REPLACE FUNCTION application.hybrid_search(
    p_embedding VECTOR(1536),
    p_keywords TEXT[] DEFAULT NULL,
    p_theme VARCHAR(500) DEFAULT NULL,
    p_doctype VARCHAR(50) DEFAULT NULL,
    p_user_id UUID DEFAULT NULL,
    p_language VARCHAR(50) DEFAULT NULL,
    p_similarity_threshold FLOAT DEFAULT 0.7,
    p_limit INTEGER DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    chunk_content TEXT,
    filename VARCHAR(500),
    chapter_name VARCHAR(500),
    doctype VARCHAR(50),
    keywords TEXT[],
    vector_similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        v.id,
        v.chunk_content,
        v.filename,
        v.chapter_name,
        v.doctype,
        v.keywords,
        (1 - (v.embeddings <=> p_embedding)) as vector_similarity
    FROM application.vectordatatable v
    WHERE (1 - (v.embeddings <=> p_embedding)) > p_similarity_threshold
    AND (p_keywords IS NULL OR v.keywords && p_keywords)
    AND (p_theme IS NULL OR v.theme ILIKE '%' || p_theme || '%')
    AND (p_doctype IS NULL OR v.doctype = p_doctype)
    AND (p_user_id IS NULL OR v.user_id = p_user_id)
    AND (p_language IS NULL OR v.content_language = p_language)
    ORDER BY v.embeddings <=> p_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Get citation from scripture (specific for Saarthi)
CREATE OR REPLACE FUNCTION application.get_scripture_citation(
    p_embedding VECTOR(1536),
    p_keywords TEXT[],
    p_doctype VARCHAR(50),
    p_language VARCHAR(50) DEFAULT 'english',
    p_limit INTEGER DEFAULT 3
)
RETURNS TABLE (
    id UUID,
    chunk_content TEXT,
    filename VARCHAR(500),
    chapter_name VARCHAR(500),
    doctype VARCHAR(50),
    similarity_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        v.id,
        v.chunk_content,
        v.filename,
        v.chapter_name,
        v.doctype,
        (1 - (v.embeddings <=> p_embedding)) as similarity_score
    FROM application.vectordatatable v
    WHERE v.doctype = p_doctype
    AND (p_keywords IS NULL OR v.keywords && p_keywords)
    AND (p_language IS NULL OR v.content_language = p_language)
    ORDER BY v.embeddings <=> p_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Get parent chunks for child results
CREATE OR REPLACE FUNCTION application.get_parent_chunks(p_child_ids UUID[])
RETURNS TABLE (
    child_id UUID,
    parent_content TEXT,
    filename VARCHAR(500)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        v.id as child_id,
        h.parent_chunk_content,
        h.filename
    FROM application.vectordatatable v
    JOIN application.hierarchytable h ON v.id = h.id
    WHERE v.id = ANY(p_child_ids);
END;
$$ LANGUAGE plpgsql;

-- Multilingual search support
CREATE OR REPLACE FUNCTION application.multilingual_search(
    p_embedding VECTOR(1536),
    p_languages TEXT[],
    p_doctype VARCHAR(50) DEFAULT NULL,
    p_limit INTEGER DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    chunk_content TEXT,
    filename VARCHAR(500),
    content_language VARCHAR(50),
    similarity_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        v.id,
        v.chunk_content,
        v.filename,
        v.content_language,
        (1 - (v.embeddings <=> p_embedding)) as similarity_score
    FROM application.vectordatatable v
    WHERE v.content_language = ANY(p_languages)
    AND (p_doctype IS NULL OR v.doctype = p_doctype)
    ORDER BY v.embeddings <=> p_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SAMPLE DATA (For Testing - Optional)
-- ============================================================================

-- Uncomment to insert sample scripture data
-- INSERT INTO application.vectordatatable (filename, doctype, chunk_content, chapter_name, keywords, content_language)
-- VALUES 
--     ('bhagavad_gita_chapter_2.txt', 'bhagavat_geeta', 
--      'You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions.',
--      'Chapter 2: Sankhya Yoga', 
--      ARRAY['karma', 'duty', 'action', 'detachment'],
--      'english'),
--     
--     ('upanishad_brihadaranyaka.txt', 'upanishad',
--      'From the unreal lead me to the real, from darkness lead me to light, from death lead me to immortality.',
--      'Brihadaranyaka Upanishad', 
--      ARRAY['truth', 'enlightenment', 'prayer', 'wisdom'],
--      'english');

-- ============================================================================
-- GRANTS
-- ============================================================================

-- Example grants (uncomment and adjust as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA application TO app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA application TO app_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA application TO app_user;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE application.vectordatatable IS 'Document chunks with embeddings for RAG retrieval and citation';
COMMENT ON TABLE application.hierarchytable IS 'Parent-child relationships for hierarchical chunking';

COMMENT ON COLUMN application.vectordatatable.doctype IS 'Scripture classification: ved, puraan, upanishad, bhagavat_geeta, arthashastra, others';
COMMENT ON COLUMN application.vectordatatable.embeddings IS '1536-dim OpenAI embedding for semantic search';
COMMENT ON COLUMN application.vectordatatable.theme_vector IS '384-dim SentenceTransformer embedding for theme matching';
COMMENT ON COLUMN application.vectordatatable.keywords IS 'Extracted keywords for hybrid search';
COMMENT ON COLUMN application.vectordatatable.content_language IS 'Language of the document content for multilingual support';