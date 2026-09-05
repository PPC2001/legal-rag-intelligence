-- ============================================================================
-- ⚖️ Legal RAG QA — Database Initialization & Schema Setup (PostgreSQL 16 + pgvector)
-- ============================================================================
-- Purpose:
--   Sets up the complete relational + vector database schema required for
--   Legal RAG QA, supporting up to 1,000,000+ documents (~20,000,000 chunks).
--
-- How to apply:
--   psql "$DATABASE_URL" -f data.sql
--   or execute directly in the Neon SQL Console / pgAdmin / DBeaver.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Enable Required Extensions
-- ----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Dedicated schema for self-hosted Langfuse tracing (prevents Prisma migration conflicts)
CREATE SCHEMA IF NOT EXISTS langfuse;

-- ----------------------------------------------------------------------------
-- 2. Collection Metadata Table (langchain-postgres)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS langchain_pg_collection (
    uuid UUID NOT NULL PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR NOT NULL,
    cmetadata JSON,
    CONSTRAINT langchain_pg_collection_name_key UNIQUE (name)
);

-- ----------------------------------------------------------------------------
-- 3. Document Embeddings & Chunk Storage Table
-- ----------------------------------------------------------------------------
-- Note: Embedding dimension is 768 to match Google Gemini embedding models
--       (models/gemini-embedding-001 / text-embedding-004).
--       tsv is an automatically computed tsvector column for zero-RAM full-text search.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    id VARCHAR NOT NULL PRIMARY KEY,
    collection_id UUID REFERENCES langchain_pg_collection (uuid) ON DELETE CASCADE,
    embedding vector(768),
    document VARCHAR,
    cmetadata JSONB,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(document, ''))) STORED
);

-- Ensure embedding dimension matches 768 if table already existed without explicit dimensions
ALTER TABLE langchain_pg_embedding 
    ALTER COLUMN embedding TYPE vector(768);

-- Ensure tsv column exists if table was created previously without generated tsvector
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'langchain_pg_embedding' AND column_name = 'tsv'
    ) THEN
        ALTER TABLE langchain_pg_embedding 
        ADD COLUMN tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(document, ''))) STORED;
    END IF;
END $$;

-- ----------------------------------------------------------------------------
-- 4. High-Performance Indexes for 1,000,000+ Document Scale
-- ----------------------------------------------------------------------------

-- A. HNSW Vector Index for Sub-50ms Approximate Nearest Neighbor (ANN) Cosine Search
--    Configured with m=16, ef_construction=64 for optimal speed/recall balance on 20M rows.
CREATE INDEX IF NOT EXISTS chunk_hnsw_idx 
ON langchain_pg_embedding 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- B. GIN Index for Native PostgreSQL Full-Text Keyword Search
--    Replaces in-memory BM25 to prevent Python RAM exhaustion at 1M documents.
CREATE INDEX IF NOT EXISTS chunk_tsv_gin_idx 
ON langchain_pg_embedding 
USING gin (tsv);

-- C. GIN Index on JSONB Metadata
--    Accelerates fast filtering by doc_hash, filename, page, or metadata tags.
CREATE INDEX IF NOT EXISTS ix_cmetadata_gin 
ON langchain_pg_embedding 
USING gin (cmetadata jsonb_path_ops);

-- D. B-Tree Index on Collection Foreign Key
CREATE INDEX IF NOT EXISTS ix_embedding_collection_id 
ON langchain_pg_embedding (collection_id);

-- ----------------------------------------------------------------------------
-- 5. Default Collection Seed
-- ----------------------------------------------------------------------------
INSERT INTO langchain_pg_collection (uuid, name, cmetadata)
VALUES (
    '63cdd374-4a93-4e14-97b5-46856e176303',
    'legal_documents',
    '{"description": "Default document collection for Legal RAG QA platform", "created_by": "system"}'::json
)
ON CONFLICT (name) DO UPDATE 
SET cmetadata = EXCLUDED.cmetadata 
WHERE langchain_pg_collection.cmetadata IS NULL;

-- ----------------------------------------------------------------------------
-- 6. Verification & Schema Status Check
-- ----------------------------------------------------------------------------
SELECT 
    c.relname AS table_name,
    i.relname AS index_name,
    am.amname AS index_algorithm
FROM pg_class c
JOIN pg_index x ON c.oid = x.indrelid
JOIN pg_class i ON i.oid = x.indexrelid
JOIN pg_am am ON i.relam = am.oid
WHERE c.relname IN ('langchain_pg_collection', 'langchain_pg_embedding')
ORDER BY c.relname, i.relname;

SELECT 
    'Collections' AS entity,
    count(*)::text AS count 
FROM langchain_pg_collection
UNION ALL
SELECT 
    'Chunks / Embeddings' AS entity,
    count(*)::text AS count 
FROM langchain_pg_embedding;
