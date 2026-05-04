CREATE TABLE edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID REFERENCES nodes(id) ON DELETE CASCADE,
    target_id UUID REFERENCES nodes(id) ON DELETE CASCADE,
    label edge_label NOT NULL,
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(source_id, target_id, label)
);
