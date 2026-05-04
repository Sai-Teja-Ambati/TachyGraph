-- ============================================================
-- TachyGraph — Unified Schema Init (merged from V1–V10)
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Enums
CREATE TYPE node_label AS ENUM (
    'PROJECT', 'SUMMARY',
    'QUESTION', 'ANSWER',
    'PREFERENCE'
);

CREATE TYPE edge_label AS ENUM (
    'PART_OF', 'SUPERSEDES',
    'ANSWERS', 'CONTEXT_OF', 'ELABORATES',
    'RELEVANT_TO'
);

-- Nodes table
CREATE TABLE nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label node_label NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1024),
    summary TEXT,
    tfidf JSONB,
    doc_length INTEGER DEFAULT 0,
    cluster_id UUID,
    confidence FLOAT CHECK (confidence BETWEEN 0 AND 1),
    degree INTEGER DEFAULT 0,
    degree_cap INTEGER DEFAULT 10,
    valid_from TIMESTAMP DEFAULT NOW(),
    valid_until TIMESTAMP,
    provenance JSONB,
    project_id UUID NOT NULL,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT chk_provenance CHECK (
        label NOT IN ('ANSWER', 'SUMMARY') OR
        (provenance->>'source_url' IS NOT NULL)
    ),
    CONSTRAINT chk_question_cap CHECK (
        label != 'QUESTION' OR degree_cap = 10
    ),
    CONSTRAINT chk_answer_cap CHECK (
        label != 'ANSWER' OR degree_cap = 10
    ),
    CONSTRAINT chk_project_exempt CHECK (
        label != 'PROJECT' OR degree_cap = 0
    ),
    CONSTRAINT chk_summary_required CHECK (
        label NOT IN ('SUMMARY', 'ANSWER')
        OR summary IS NOT NULL
    ),
    CONSTRAINT chk_embedding_on_searchable CHECK (
        label NOT IN ('SUMMARY', 'QUESTION', 'ANSWER')
        OR embedding IS NOT NULL
    )
);

-- BM25 tables
CREATE TABLE bm25_df (
    term TEXT PRIMARY KEY,
    doc_count INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE bm25_stats (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    total_docs INTEGER NOT NULL DEFAULT 0,
    avg_doc_length FLOAT NOT NULL DEFAULT 0
);

INSERT INTO bm25_stats (id, total_docs, avg_doc_length) VALUES (1, 0, 0);

-- Edges table
CREATE TABLE edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID REFERENCES nodes(id) ON DELETE CASCADE,
    target_id UUID REFERENCES nodes(id) ON DELETE CASCADE,
    label edge_label NOT NULL,
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(source_id, target_id, label)
);

-- FAISS sync state
CREATE TABLE faiss_sync_state (
    project_id UUID PRIMARY KEY,
    last_sync_at TIMESTAMP DEFAULT NOW(),
    vector_count INTEGER DEFAULT 0,
    faiss_index_path TEXT,
    is_trained BOOLEAN DEFAULT FALSE
);

-- Tasks / reminders
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    description TEXT NOT NULL,
    due_at TIMESTAMP NOT NULL,
    completed BOOLEAN DEFAULT FALSE,
    completed_at TIMESTAMP,
    project_id UUID NOT NULL,
    related_node_id UUID REFERENCES nodes(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chat sessions (persistent)
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID,
    created_at TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP DEFAULT NOW()
);

CREATE TABLE session_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_session_messages_session ON session_messages (session_id, created_at);

-- ============================================================
-- Triggers
-- ============================================================

-- A. Block QUESTION → QUESTION edges
CREATE OR REPLACE FUNCTION block_qq_links() RETURNS TRIGGER AS $$
BEGIN
    IF (SELECT label FROM nodes WHERE id = NEW.source_id) = 'QUESTION'
       AND (SELECT label FROM nodes WHERE id = NEW.target_id) = 'QUESTION' THEN
        RAISE EXCEPTION 'Questions cannot connect to other questions';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_01_block_qq
BEFORE INSERT ON edges
FOR EACH ROW
EXECUTE FUNCTION block_qq_links();

-- B. 5-Slot answer eviction with edge migration
CREATE OR REPLACE FUNCTION enforce_answer_slots() RETURNS TRIGGER AS $$
DECLARE
    min_conf FLOAT;
    old_answer_id UUID;
    new_conf FLOAT;
BEGIN
    IF NEW.label != 'ANSWERS' THEN RETURN NEW; END IF;

    IF (SELECT COUNT(*) FROM edges WHERE target_id = NEW.target_id AND label = 'ANSWERS') < 10 THEN
        UPDATE nodes SET degree = degree + 1 WHERE id = NEW.target_id;
        RETURN NEW;
    END IF;

    SELECT a.id, a.confidence INTO old_answer_id, min_conf
    FROM edges e JOIN nodes a ON e.source_id = a.id
    WHERE e.target_id = NEW.target_id AND e.label = 'ANSWERS'
    ORDER BY a.confidence ASC, a.created_at ASC LIMIT 1;

    SELECT confidence INTO new_conf FROM nodes WHERE id = NEW.source_id;

    IF new_conf <= min_conf THEN
        RAISE EXCEPTION 'Answer rejected: confidence % <= %', new_conf, min_conf;
    END IF;

    UPDATE nodes SET valid_until = NOW() WHERE id = old_answer_id;

    INSERT INTO edges (source_id, target_id, label)
    VALUES (NEW.source_id, old_answer_id, 'SUPERSEDES');

    UPDATE edges SET source_id = NEW.source_id
    WHERE source_id = old_answer_id AND label IN ('CONTEXT_OF', 'ELABORATES');

    UPDATE edges SET target_id = NEW.source_id
    WHERE target_id = old_answer_id AND label IN ('CONTEXT_OF', 'ELABORATES');

    DELETE FROM edges
    WHERE source_id = old_answer_id AND target_id = NEW.target_id AND label = 'ANSWERS';

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_02_answer_slots
BEFORE INSERT ON edges
FOR EACH ROW
EXECUTE FUNCTION enforce_answer_slots();

-- C. General degree cap (PROJECT nodes exempted via degree_cap = 0)
CREATE OR REPLACE FUNCTION enforce_degree_cap() RETURNS TRIGGER AS $$
DECLARE
    src_cap INTEGER;
    src_deg INTEGER;
    tgt_cap INTEGER;
    tgt_deg INTEGER;
BEGIN
    SELECT degree_cap, degree INTO src_cap, src_deg FROM nodes WHERE id = NEW.source_id;
    SELECT degree_cap, degree INTO tgt_cap, tgt_deg FROM nodes WHERE id = NEW.target_id;

    IF src_cap > 0 AND src_deg >= src_cap THEN
        RAISE EXCEPTION 'Source % at degree cap %', NEW.source_id, src_cap;
    END IF;

    IF NEW.label != 'ANSWERS' AND tgt_cap > 0 AND tgt_deg >= tgt_cap THEN
        RAISE EXCEPTION 'Target % at degree cap %', NEW.target_id, tgt_cap;
    END IF;

    UPDATE nodes SET degree = degree + 1 WHERE id = NEW.source_id;
    IF NEW.label != 'ANSWERS' THEN
        UPDATE nodes SET degree = degree + 1 WHERE id = NEW.target_id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_03_degree_cap
BEFORE INSERT ON edges
FOR EACH ROW
EXECUTE FUNCTION enforce_degree_cap();

-- D. Temporal window — 5 days for Q&A nodes, 365 days for SUMMARY, none for PROJECT
CREATE OR REPLACE FUNCTION set_temporal_window()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.valid_from IS NOT NULL AND NEW.valid_until IS NULL THEN
        IF NEW.label = 'SUMMARY' THEN
            NEW.valid_until := NEW.valid_from + INTERVAL '365 days';
        ELSIF NEW.label IN ('PROJECT', 'PREFERENCE') THEN
            NEW.valid_until := NULL;
        ELSE
            NEW.valid_until := NEW.valid_from + INTERVAL '5 days';
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_temporal_window
BEFORE INSERT ON nodes
FOR EACH ROW EXECUTE FUNCTION set_temporal_window();

-- ============================================================
-- Functions
-- ============================================================

-- Exponential temporal decay
CREATE OR REPLACE FUNCTION exp_decay(start_ts TIMESTAMP) RETURNS FLOAT AS $$
BEGIN
    RETURN EXP(-0.1 * EXTRACT(EPOCH FROM (NOW() - start_ts)) / 86400);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- BM25 scoring
CREATE OR REPLACE FUNCTION bm25_score(
    query_tf JSONB,
    doc_tf JSONB,
    doc_len INTEGER
) RETURNS FLOAT AS $$
DECLARE
    score FLOAT := 0;
    qterm TEXT;
    tf FLOAT;
    df_val FLOAT;
    idf FLOAT;
    total_docs FLOAT;
    avgdl FLOAT;
    k1 FLOAT := 1.2;
    b FLOAT := 0.75;
BEGIN
    IF query_tf IS NULL OR doc_tf IS NULL THEN RETURN 0; END IF;

    SELECT COALESCE(s.total_docs, 1), COALESCE(s.avg_doc_length, 1)
      INTO total_docs, avgdl
      FROM bm25_stats s
     WHERE s.id = 1;

    IF total_docs IS NULL THEN
        total_docs := 1;
        avgdl := 1;
    END IF;

    FOR qterm IN SELECT jsonb_object_keys(query_tf) INTERSECT SELECT jsonb_object_keys(doc_tf)
    LOOP
        tf := (doc_tf->>qterm)::FLOAT;
        SELECT COALESCE(d.doc_count, 0) INTO df_val FROM bm25_df d WHERE d.term = qterm;
        IF df_val IS NULL THEN df_val := 0; END IF;

        idf := LN((total_docs - df_val + 0.5) / (df_val + 0.5) + 1);
        score := score + idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl));
    END LOOP;

    RETURN score;
END;
$$ LANGUAGE plpgsql STABLE;

-- Extend validity
CREATE OR REPLACE FUNCTION extend_validity(target_node_id UUID, extension_days INT DEFAULT 5)
RETURNS VOID AS $$
BEGIN
    UPDATE nodes
    SET valid_until = GREATEST(valid_until, NOW() + (extension_days || ' days')::INTERVAL)
    WHERE id = target_node_id;
END;
$$ LANGUAGE plpgsql;

-- Check if node is expiring soon
CREATE OR REPLACE FUNCTION is_expiring_soon(target_node_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    node_valid_until TIMESTAMP;
BEGIN
    SELECT valid_until INTO node_valid_until FROM nodes WHERE id = target_node_id;
    RETURN node_valid_until IS NOT NULL
       AND node_valid_until > NOW()
       AND node_valid_until <= NOW() + INTERVAL '24 hours';
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- Indexes
-- ============================================================

CREATE INDEX idx_nodes_embedding ON nodes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_nodes_project ON nodes (project_id);
CREATE INDEX idx_nodes_temporal ON nodes (valid_from, valid_until);
CREATE INDEX idx_nodes_tfidf ON nodes USING gin (tfidf jsonb_path_ops);
CREATE INDEX idx_nodes_label ON nodes (label);
CREATE INDEX idx_nodes_search ON nodes (project_id, label) WHERE embedding IS NOT NULL;
CREATE INDEX idx_nodes_faiss_sync ON nodes (project_id, label, valid_until) WHERE embedding IS NOT NULL;
