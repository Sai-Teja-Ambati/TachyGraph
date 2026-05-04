from __future__ import annotations

import uuid

import numpy as np
import psycopg.types.json

from tachyrag.config import SEARCH_K, SEARCH_BM25_WEIGHT, SEARCH_VECTOR_WEIGHT, SEARCH_TEMPORAL_WEIGHT
from tachyrag.core.db import pool
from tachyrag.core.embedder import embed
from tachyrag.faiss.vector_index import UnifiedVectorIndex
from tachyrag.core.tfidf import compute_tf


def search_fast(
    faiss_index: UnifiedVectorIndex,
    query_text: str,
    project_id: uuid.UUID,
    k: int = SEARCH_K,
    nprobe: int = 64,
) -> list[dict]:
    """FAISS GPU search: <5ms for 100M vectors. Returns nodes from pgvector by UUID."""
    query_embedding = np.array(embed(query_text), dtype="float32")
    distances, uuids = faiss_index.search(query_embedding, k, nprobe)

    valid_uuids = [u for u in uuids if u is not None]
    if not valid_uuids:
        return []

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM nodes
            WHERE id = ANY(%s)
              AND project_id = %s
              AND (valid_until IS NULL OR valid_until > NOW())
            """,
            (valid_uuids, project_id),
        )
        rows = cur.fetchall()

    node_map = {str(r["id"]): r for r in rows}
    results = []
    for uid, dist in zip(uuids, distances):
        if uid and uid in node_map:
            node = node_map[uid]
            node["faiss_score"] = float(dist)
            node["search_type"] = "faiss"
            results.append(node)
    return results


def search_hybrid(
    faiss_index: UnifiedVectorIndex,
    query_text: str,
    project_id: uuid.UUID,
    k: int = SEARCH_K,
) -> list[dict]:
    """Two-stage: FAISS GPU recall (top 100) → pgvector re-rank with multi-signal."""
    query_embedding = np.array(embed(query_text), dtype="float32")
    query_tf, _ = compute_tf(query_text)

    # Stage 1: FAISS fast recall
    distances, uuids = faiss_index.search(query_embedding, k=100, nprobe=128)
    valid_uuids = [u for u in uuids if u is not None]

    if not valid_uuids:
        return []

    # Stage 2: Re-rank candidates with configurable multi-signal scoring
    bw, vw, tw = SEARCH_BM25_WEIGHT, SEARCH_VECTOR_WEIGHT, SEARCH_TEMPORAL_WEIGHT
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT n.*,
                (
                    %s * bm25_score(%s::jsonb, n.tfidf, n.doc_length) +
                    %s * (1 - (n.embedding <=> %s::vector)) +
                    %s * exp_decay(n.valid_from)
                ) AS rank
            FROM nodes n
            WHERE n.id = ANY(%s)
              AND n.project_id = %s
              AND (n.valid_until IS NULL OR n.valid_until > NOW())
            ORDER BY rank DESC
            LIMIT %s
            """,
            (
                bw, psycopg.types.json.Jsonb(query_tf),
                vw, query_embedding.tolist(),
                tw,
                valid_uuids, project_id, k,
            ),
        )
        results = cur.fetchall()

    for r in results:
        r["search_type"] = "hybrid"
    return results
