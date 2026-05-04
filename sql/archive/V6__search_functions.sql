-- Exponential temporal decay: 30-day-old fact ≈ 0.05 weight
CREATE OR REPLACE FUNCTION exp_decay(start_ts TIMESTAMP) RETURNS FLOAT AS $$
BEGIN
    RETURN EXP(-0.1 * EXTRACT(EPOCH FROM (NOW() - start_ts)) / 86400);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- BM25 score: query_tf JSONB × doc_tf JSONB, using doc_length and global stats
-- k1=1.2, b=0.75 (Lucene/ES defaults)
-- IDF(t) = ln((N - df + 0.5) / (df + 0.5) + 1)
-- score  = Σ IDF(t) × (tf × (k1+1)) / (tf + k1 × (1 - b + b × dl/avgdl))
CREATE OR REPLACE FUNCTION bm25_score(
    query_tf JSONB,
    doc_tf JSONB,
    doc_len INTEGER
) RETURNS FLOAT AS $$
DECLARE
    score FLOAT := 0;
    term TEXT;
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

    FOR term IN SELECT jsonb_object_keys(query_tf) INTERSECT SELECT jsonb_object_keys(doc_tf)
    LOOP
        tf := (doc_tf->>term)::FLOAT;
        SELECT COALESCE(d.doc_count, 0) INTO df_val FROM bm25_df d WHERE d.term = term;
        IF df_val IS NULL THEN df_val := 0; END IF;

        idf := LN((total_docs - df_val + 0.5) / (df_val + 0.5) + 1);
        score := score + idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl));
    END LOOP;

    RETURN score;
END;
$$ LANGUAGE plpgsql STABLE;
