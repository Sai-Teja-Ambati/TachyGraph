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
        label NOT IN ('ANSWER', 'CHUNK') OR
        (provenance->>'source_url' IS NOT NULL)
    ),
    CONSTRAINT chk_question_cap CHECK (
        label != 'QUESTION' OR degree_cap = 5
    ),
    CONSTRAINT chk_answer_cap CHECK (
        label != 'ANSWER' OR degree_cap = 10
    ),
    CONSTRAINT chk_project_exempt CHECK (
        label != 'PROJECT' OR degree_cap = 0
    )
);

-- BM25 global document frequency: how many documents contain each term
CREATE TABLE bm25_df (
    term TEXT PRIMARY KEY,
    doc_count INTEGER NOT NULL DEFAULT 1
);

-- BM25 corpus stats: single-row table for total docs and average doc length
CREATE TABLE bm25_stats (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    total_docs INTEGER NOT NULL DEFAULT 0,
    avg_doc_length FLOAT NOT NULL DEFAULT 0
);

INSERT INTO bm25_stats (id, total_docs, avg_doc_length) VALUES (1, 0, 0);
