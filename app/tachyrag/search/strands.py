from __future__ import annotations

import uuid
from enum import Enum

import psycopg.types.json

from tachyrag.config import SEARCH_K, SEARCH_BM25_WEIGHT, SEARCH_VECTOR_WEIGHT, SEARCH_TEMPORAL_WEIGHT
from tachyrag.core.db import pool
from tachyrag.core.embedder import embed
from tachyrag.core.tfidf import compute_tf


class SearchStrand(Enum):
    EXACT_MATCH = "exact"
    CONTEXT_WEAVE = "weave"
    TEMPORAL_DEEP = "temporal"
    SEMANTIC_NEAR = "semantic"


INTENT_STRAND_MAP = {
    "debugging": [SearchStrand.EXACT_MATCH, SearchStrand.CONTEXT_WEAVE],
    "reference": [SearchStrand.EXACT_MATCH, SearchStrand.SEMANTIC_NEAR],
    "comparison": [SearchStrand.TEMPORAL_DEEP, SearchStrand.EXACT_MATCH],
    "history": [SearchStrand.TEMPORAL_DEEP],
}


def select_strands(intent: dict) -> list[SearchStrand]:
    intent_type = intent.get("intent", "reference")
    return INTENT_STRAND_MAP.get(intent_type, [SearchStrand.EXACT_MATCH])


def run_strand_exact(
    query_embedding: list[float],
    query_tf: dict,
    project_id: uuid.UUID,
    k: int = SEARCH_K,
) -> list[dict]:
    candidate_pool = k * 5
    bw, vw, tw = SEARCH_BM25_WEIGHT, SEARCH_VECTOR_WEIGHT, SEARCH_TEMPORAL_WEIGHT
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH candidates AS (
                SELECT n.id
                FROM nodes n
                WHERE n.project_id = %s
                  AND (n.valid_until IS NULL OR n.valid_until > NOW())
                  AND n.embedding IS NOT NULL
                ORDER BY n.embedding <=> %s::vector
                LIMIT %s
            )
            SELECT n.*, 'exact' AS strand,
                (
                    %s * bm25_score(%s::jsonb, n.tfidf, n.doc_length) +
                    %s * (1 - (n.embedding <=> %s::vector)) +
                    %s * exp_decay(n.valid_from)
                ) AS rank
            FROM nodes n
            JOIN candidates c ON c.id = n.id
            ORDER BY rank DESC
            LIMIT %s
            """,
            (project_id, query_embedding, candidate_pool,
             bw, psycopg.types.json.Jsonb(query_tf), vw, query_embedding, tw, k),
        )
        return cur.fetchall()


def run_strand_weave(
    seed_node_ids: list[uuid.UUID],
    k: int = 7,
) -> list[dict]:
    if not seed_node_ids:
        return []
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH RECURSIVE hops AS (
                SELECT e.target_id AS id, e.label AS rel, 1 AS depth
                FROM edges e
                WHERE e.source_id = ANY(%s)
                  AND e.label IN ('CONTEXT_OF', 'ELABORATES')
                UNION
                SELECT e2.target_id, e2.label, h.depth + 1
                FROM hops h
                JOIN edges e2 ON e2.source_id = h.id
                WHERE h.depth < 2
                  AND e2.label IN ('CONTEXT_OF', 'ELABORATES')
            )
            SELECT DISTINCT n.*, 'weave' AS strand, h.rel, h.depth
            FROM hops h
            JOIN nodes n ON n.id = h.id
            WHERE (n.valid_until IS NULL OR n.valid_until > NOW())
            LIMIT %s
            """,
            (seed_node_ids, k),
        )
        return cur.fetchall()


def run_strand_temporal(
    query_embedding: list[float],
    project_id: uuid.UUID,
    k: int = 7,
) -> list[dict]:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH current_facts AS (
                SELECT id FROM nodes
                WHERE project_id = %s
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 5
            ),
            superseded AS (
                SELECT e.target_id AS id, e.label AS rel
                FROM edges e
                JOIN current_facts cf ON cf.id = e.source_id
                WHERE e.label = 'SUPERSEDES'
            )
            SELECT n.*, 'temporal' AS strand
            FROM superseded s
            JOIN nodes n ON n.id = s.id
            LIMIT %s
            """,
            (project_id, query_embedding, k),
        )
        return cur.fetchall()


def run_strand_semantic(
    query_embedding: list[float],
    project_id: uuid.UUID,
    k: int = 7,
) -> list[dict]:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT n.*, 'semantic' AS strand,
                   (1 - (n.embedding <=> %s::vector)) AS rank
            FROM nodes n
            WHERE n.project_id = %s
              AND (n.valid_until IS NULL OR n.valid_until > NOW())
              AND n.embedding IS NOT NULL
            ORDER BY n.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, project_id, query_embedding, k),
        )
        return cur.fetchall()


def execute_strands(
    intent: dict,
    query_text: str,
    project_id: uuid.UUID,
    k: int = SEARCH_K,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    strands = select_strands(intent)
    rephrased = intent.get("rephrased", query_text)
    if query_embedding is None:
        query_embedding = embed(rephrased)
    query_tf, _ = compute_tf(rephrased)

    per_strand_k = max(k // len(strands), 3) if strands else k
    all_results = []
    seen_ids = set()

    for strand in strands:
        if strand == SearchStrand.EXACT_MATCH:
            rows = run_strand_exact(query_embedding, query_tf, project_id, per_strand_k)
        elif strand == SearchStrand.CONTEXT_WEAVE:
            seed_ids = [r["id"] for r in all_results[:3]]
            rows = run_strand_weave(seed_ids, per_strand_k)
        elif strand == SearchStrand.TEMPORAL_DEEP:
            rows = run_strand_temporal(query_embedding, project_id, per_strand_k)
        elif strand == SearchStrand.SEMANTIC_NEAR:
            rows = run_strand_semantic(query_embedding, project_id, per_strand_k)
        else:
            continue

        for r in rows:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                all_results.append(r)

    return all_results[:k]
