-- HNSW for vector cosine similarity search
CREATE INDEX idx_nodes_embedding ON nodes USING hnsw (embedding vector_cosine_ops);

-- B-tree for project scoping
CREATE INDEX idx_nodes_project ON nodes (project_id);

-- Composite temporal range queries
CREATE INDEX idx_nodes_temporal ON nodes (valid_from, valid_until);

-- GIN for TF-IDF JSONB containment (@>) queries
CREATE INDEX idx_nodes_tfidf ON nodes USING gin (tfidf jsonb_path_ops);

-- B-tree for label-filtered scans
CREATE INDEX idx_nodes_label ON nodes (label);

-- Trigram for fuzzy text search on content
CREATE INDEX idx_nodes_content_trgm ON nodes USING gin (content gin_trgm_ops);
