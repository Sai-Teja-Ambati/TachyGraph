from __future__ import annotations

import uuid

import psycopg.types.json

from tachyrag.config import SEARCH_K, SEARCH_BM25_WEIGHT, SEARCH_VECTOR_WEIGHT, SEARCH_TEMPORAL_WEIGHT, SEARCH_USE_DECOMPOSITION
from tachyrag.core.db import pool
from tachyrag.search.disambiguator import OllamaDisambiguator
from tachyrag.core.embedder import embed
from tachyrag.core.tfidf import compute_tf
from tachyrag.graph.temporal import bump_access_and_reaffirm

_disambiguator: OllamaDisambiguator | None = None


def _get_disambiguator() -> OllamaDisambiguator:
    global _disambiguator
    if _disambiguator is None:
        _disambiguator = OllamaDisambiguator()
    return _disambiguator


def estimate_k(query: str) -> int:
    words = query.lower().split()
    if len(words) <= 2 or any(w in words for w in ["hi", "hello", "hey", "thanks", "bye"]):
        return 0
    if len(words) <= 4:
        return 3
    if any(w in words for w in ["compare", "vs", "versus", "difference", "between"]):
        return 15
    if any(w in words for w in ["explain", "how", "architecture", "overview", "all"]):
        return 10
    return 7


def _resolve_weights(
    bm25_weight: float | None,
    vector_weight: float | None,
    temporal_weight: float | None,
) -> tuple[float, float, float]:
    """Use per-request weights if all three provided, else fall back to config defaults."""
    if bm25_weight is not None and vector_weight is not None and temporal_weight is not None:
        return bm25_weight, vector_weight, temporal_weight
    return SEARCH_BM25_WEIGHT, SEARCH_VECTOR_WEIGHT, SEARCH_TEMPORAL_WEIGHT


def _search_core(
    query_embedding: list[float],
    query_tf: dict,
    project_id: uuid.UUID | None,
    k: int,
    bw: float,
    vw: float,
    tw: float,
) -> list[dict]:
    candidate_pool = k * 5
    with pool.connection() as conn, conn.cursor() as cur:
        if project_id:
            cur.execute(
                """
                WITH candidates AS (
                    SELECT n.id FROM nodes n
                    WHERE n.project_id = %(pid)s
                      AND n.label IN ('SUMMARY', 'ANSWER')
                      AND (n.valid_until IS NULL OR n.valid_until > NOW())
                      AND n.embedding IS NOT NULL
                    ORDER BY n.embedding <=> %(emb)s::vector
                    LIMIT %(pool)s
                )
                SELECT n.*,
                    (%(bw)s * bm25_score(%(qtf)s::jsonb, n.tfidf, n.doc_length) +
                     %(vw)s * (1 - (n.embedding <=> %(emb)s::vector)) +
                     %(tw)s * exp_decay(n.valid_from)) AS rank
                FROM nodes n JOIN candidates c ON c.id = n.id
                ORDER BY rank DESC LIMIT %(k)s
                """,
                {"qtf": psycopg.types.json.Jsonb(query_tf), "emb": query_embedding, "pid": project_id,
                 "pool": candidate_pool, "k": k, "bw": bw, "vw": vw, "tw": tw},
            )
        else:
            cur.execute(
                """
                WITH candidates AS (
                    SELECT n.id FROM nodes n
                    WHERE n.label IN ('SUMMARY', 'ANSWER')
                      AND (n.valid_until IS NULL OR n.valid_until > NOW())
                      AND n.embedding IS NOT NULL
                    ORDER BY n.embedding <=> %(emb)s::vector
                    LIMIT %(pool)s
                )
                SELECT n.*,
                    (%(bw)s * bm25_score(%(qtf)s::jsonb, n.tfidf, n.doc_length) +
                     %(vw)s * (1 - (n.embedding <=> %(emb)s::vector)) +
                     %(tw)s * exp_decay(n.valid_from)) AS rank
                FROM nodes n JOIN candidates c ON c.id = n.id
                ORDER BY rank DESC LIMIT %(k)s
                """,
                {"qtf": psycopg.types.json.Jsonb(query_tf), "emb": query_embedding,
                 "pool": candidate_pool, "k": k, "bw": bw, "vw": vw, "tw": tw},
            )
        return cur.fetchall()


def search(
    query_text: str,
    project_id: uuid.UUID | None = None,
    k: int = SEARCH_K,
    use_hyde: bool = False,
    bm25_weight: float | None = None,
    vector_weight: float | None = None,
    temporal_weight: float | None = None,
) -> list[dict]:
    """Search with optional per-request weight overrides."""
    bw, vw, tw = _resolve_weights(bm25_weight, vector_weight, temporal_weight)

    if use_hyde:
        from tachyrag.search.hyde import hyde_embed
        query_embedding = hyde_embed(query_text)
    else:
        query_embedding = embed(query_text)
    query_tf, _ = compute_tf(query_text)

    if SEARCH_USE_DECOMPOSITION:
        from tachyrag.search.decomposer import decompose_query
        sub_queries = decompose_query(query_text)
    else:
        sub_queries = [query_text]

    if len(sub_queries) <= 1:
        results = _search_core(query_embedding, query_tf, project_id, k, bw, vw, tw)
    else:
        from tachyrag.search.rrf import reciprocal_rank_fusion
        ranked_lists = []
        for sq in sub_queries:
            sq_emb = embed(sq)
            sq_tf, _ = compute_tf(sq)
            ranked_lists.append(_search_core(sq_emb, sq_tf, project_id, k, bw, vw, tw))
        results = reciprocal_rank_fusion(ranked_lists, top_n=k)

    if results:
        bump_access_and_reaffirm([r["id"] for r in results])

    return results


def deep_search(
    raw_query: str,
    project_id: uuid.UUID | None = None,
    context: dict | None = None,
    k: int = SEARCH_K,
) -> dict:
    disambiguator = _get_disambiguator()
    intent = disambiguator.disambiguate(raw_query, context)

    from tachyrag.search.strands import execute_strands
    results = execute_strands(intent, raw_query, project_id, k)

    return {
        "intent": intent,
        "results": results,
        "count": len(results),
    }
