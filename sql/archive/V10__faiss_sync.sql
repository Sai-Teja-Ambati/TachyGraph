-- Track FAISS sync state per project
CREATE TABLE faiss_sync_state (
    project_id UUID PRIMARY KEY,
    last_sync_at TIMESTAMP DEFAULT NOW(),
    vector_count INTEGER DEFAULT 0,
    faiss_index_path TEXT,
    is_trained BOOLEAN DEFAULT FALSE
);

-- Index for fast sync queries (nodes with embeddings, filtered by project + validity)
CREATE INDEX idx_nodes_faiss_sync ON nodes (project_id, label, valid_until)
WHERE embedding IS NOT NULL;
