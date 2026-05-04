from __future__ import annotations

import logging
import uuid

from tachyrag.core.db import pool, insert_edge

log = logging.getLogger(__name__)


def compact_project(project_id: uuid.UUID, similarity_threshold: float = 0.98) -> dict:
    """Find near-duplicate SUMMARY nodes in a project and merge them."""
    merged = 0
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT s1.id AS keep_id, s2.id AS remove_id,
                   1 - (s1.embedding <=> s2.embedding) AS sim
            FROM nodes s1, nodes s2
            WHERE s1.label = 'SUMMARY' AND s2.label = 'SUMMARY'
              AND s1.project_id = %s AND s2.project_id = %s
              AND s1.id < s2.id
              AND s1.provenance->>'source_url' = s2.provenance->>'source_url'
              AND 1 - (s1.embedding <=> s2.embedding) > %s
              AND (s1.valid_until IS NULL OR s1.valid_until > NOW())
              AND (s2.valid_until IS NULL OR s2.valid_until > NOW())
            ORDER BY sim DESC
            LIMIT 100
            """,
            (project_id, project_id, similarity_threshold),
        )
        duplicates = cur.fetchall()

    expired_ids = set()
    for dup in duplicates:
        if dup["remove_id"] in expired_ids:
            continue
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute("UPDATE nodes SET valid_until = NOW() WHERE id = %s", (dup["remove_id"],))
            conn.commit()
        try:
            insert_edge(dup["keep_id"], dup["remove_id"], "SUPERSEDES")
        except Exception:
            pass
        expired_ids.add(dup["remove_id"])
        merged += 1

    log.info("Compacted project %s: %d duplicates merged", project_id, merged)
    return {"project_id": str(project_id), "merged": merged}


def get_expiry_report(project_id: uuid.UUID | None = None) -> dict:
    """Get a summary of expiring facts across the graph."""
    with pool.connection() as conn, conn.cursor() as cur:
        where = "WHERE project_id = %s" if project_id else ""
        params = (project_id,) if project_id else ()

        cur.execute(f"SELECT COUNT(*) AS total FROM nodes {where}", params)
        total = cur.fetchone()["total"]

        where_exp = f"{'AND' if project_id else 'WHERE'} valid_until BETWEEN NOW() AND NOW() + INTERVAL '24 hours'"
        full_where = f"WHERE project_id = %s {where_exp}" if project_id else where_exp
        cur.execute(f"SELECT COUNT(*) AS expiring_24h FROM nodes {full_where}", params)
        exp_24 = cur.fetchone()["expiring_24h"]

        where_exp7 = f"{'AND' if project_id else 'WHERE'} valid_until BETWEEN NOW() AND NOW() + INTERVAL '7 days'"
        full_where7 = f"WHERE project_id = %s {where_exp7}" if project_id else where_exp7
        cur.execute(f"SELECT COUNT(*) AS expiring_7d FROM nodes {full_where7}", params)
        exp_7d = cur.fetchone()["expiring_7d"]

        # High-value facts expiring (access_count > 3)
        where_hot = f"{'AND' if project_id else 'WHERE'} valid_until BETWEEN NOW() AND NOW() + INTERVAL '48 hours' AND access_count > 3"
        full_hot = f"WHERE project_id = %s {where_hot}" if project_id else where_hot
        cur.execute(
            f"SELECT id, summary, access_count, valid_until FROM nodes {full_hot} ORDER BY access_count DESC LIMIT 10",
            params,
        )
        hot_expiring = cur.fetchall()

    return {
        "total_nodes": total,
        "expiring_24h": exp_24,
        "expiring_7d": exp_7d,
        "hot_expiring": [
            {"id": str(r["id"]), "summary": r.get("summary", "")[:150], "access_count": r["access_count"], "valid_until": str(r["valid_until"])}
            for r in hot_expiring
        ],
    }
