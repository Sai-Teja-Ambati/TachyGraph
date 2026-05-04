from __future__ import annotations

import logging
import uuid

import numpy as np
import psycopg

from tachyrag.config import MMR_EDGE_TOP_K, MMR_LAMBDA
from tachyrag.core.db import pool

log = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0


def mmr_link_summary(
    node_id: uuid.UUID,
    embedding: list[float],
    project_id: uuid.UUID,
    top_k: int = MMR_EDGE_TOP_K,
) -> int:
    """Create RELEVANT_TO edges from a new SUMMARY node to its top-k MMR-scored peers."""
    vec = np.array(embedding, dtype=np.float32)

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, embedding FROM nodes WHERE label = 'SUMMARY' AND project_id = %s AND id != %s AND embedding IS NOT NULL",
            (project_id, node_id),
        )
        candidates = cur.fetchall()

    if not candidates:
        return 0

    # Score each candidate via MMR
    selected_vecs: list[np.ndarray] = []
    scored = []
    for c in candidates:
        c_vec = np.array(c["embedding"], dtype=np.float32)
        relevance = _cosine(vec, c_vec)
        redundancy = max((_cosine(c_vec, sv) for sv in selected_vecs), default=0.0)
        score = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * redundancy
        scored.append((c["id"], score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    created = 0
    for target_id, weight in top:
        if weight <= 0:
            continue
        try:
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO edges (source_id, target_id, label, weight) VALUES (%s, %s, 'RELEVANT_TO', %s) ON CONFLICT DO NOTHING",
                    (node_id, target_id, round(weight, 4)),
                )
                conn.commit()
            created += 1
        except (psycopg.errors.RaiseException, psycopg.errors.UniqueViolation):
            pass
    log.info("MMR linked node %s → %d edges", node_id, created)
    return created


def recompute_project_edges(project_id: uuid.UUID) -> dict:
    """Delete all RELEVANT_TO edges for a project and recompute via MMR."""
    with pool.connection() as conn, conn.cursor() as cur:
        # Delete existing RELEVANT_TO edges in this project
        cur.execute(
            """
            DELETE FROM edges WHERE label = 'RELEVANT_TO'
            AND source_id IN (SELECT id FROM nodes WHERE project_id = %s)
            """,
            (project_id,),
        )
        deleted = cur.rowcount
        conn.commit()

        # Fetch all SUMMARY nodes with embeddings
        cur.execute(
            "SELECT id, embedding FROM nodes WHERE label = 'SUMMARY' AND project_id = %s AND embedding IS NOT NULL",
            (project_id,),
        )
        nodes = cur.fetchall()

    if len(nodes) < 2:
        return {"deleted": deleted, "created": 0, "nodes": len(nodes)}

    # Precompute all embeddings as numpy
    ids = [n["id"] for n in nodes]
    vecs = [np.array(n["embedding"], dtype=np.float32) for n in nodes]

    created = 0
    for i, (nid, vec) in enumerate(zip(ids, vecs)):
        scored = []
        for j, (cid, cvec) in enumerate(zip(ids, vecs)):
            if i == j:
                continue
            relevance = _cosine(vec, cvec)
            # Redundancy against already-selected for THIS node
            scored.append((cid, relevance))

        # Apply MMR re-ranking
        selected_vecs: list[np.ndarray] = []
        mmr_ranked = []
        remaining = list(scored)
        for _ in range(min(MMR_EDGE_TOP_K, len(remaining))):
            best_idx, best_score = -1, -float("inf")
            for k, (cid, rel) in enumerate(remaining):
                cvec = vecs[ids.index(cid)]
                redundancy = max((_cosine(cvec, sv) for sv in selected_vecs), default=0.0)
                ms = MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * redundancy
                if ms > best_score:
                    best_score = ms
                    best_idx = k
            if best_idx < 0 or best_score <= 0:
                break
            cid, _ = remaining.pop(best_idx)
            selected_vecs.append(vecs[ids.index(cid)])
            mmr_ranked.append((cid, best_score))

        for target_id, weight in mmr_ranked:
            try:
                with pool.connection() as conn, conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO edges (source_id, target_id, label, weight) VALUES (%s, %s, 'RELEVANT_TO', %s) ON CONFLICT DO NOTHING",
                        (nid, target_id, round(weight, 4)),
                    )
                    conn.commit()
                created += 1
            except (psycopg.errors.RaiseException, psycopg.errors.UniqueViolation):
                pass

    log.info("MMR recompute for project %s: deleted=%d, created=%d, nodes=%d", project_id, deleted, created, len(nodes))
    return {"deleted": deleted, "created": created, "nodes": len(nodes)}


def recompute_all_projects() -> list[dict]:
    """Recompute MMR edges for every project. Used by cron."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM nodes WHERE label = 'PROJECT'")
        projects = [r["id"] for r in cur.fetchall()]

    results = []
    for pid in projects:
        try:
            r = recompute_project_edges(pid)
            r["project_id"] = str(pid)
            results.append(r)
        except Exception as e:
            log.error("MMR recompute failed for project %s: %s", pid, e)
            results.append({"project_id": str(pid), "error": str(e)})
    return results
