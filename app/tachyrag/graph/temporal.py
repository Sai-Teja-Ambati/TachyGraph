"""Temporal management: validity windows, reaffirmation, expiry, conflict resolution.

Temporal in TachyGraph means:
  - LLM extracts dates from content at ingestion time (valid_from)
  - exp_decay(valid_from) scores recent facts higher in search
  - valid_until auto-expires nodes (365d SUMMARY, 5d Q&A)
  - reaffirm extends validity when facts are re-accessed
  - Conflicts resolved by recency (newer valid_from wins), then confidence
"""
from __future__ import annotations

import logging
import uuid

from tachyrag.core.db import insert_edge, pool

log = logging.getLogger(__name__)


def reaffirm_fact(node_id: uuid.UUID, extension_days: int = 5) -> None:
    """Extend validity when a fact is re-asked / reaffirmed."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT extend_validity(%s, %s)", (node_id, extension_days))
        conn.commit()


def bump_access_and_reaffirm(node_ids: list[uuid.UUID], reaffirm_threshold: int = 5) -> int:
    """Increment access_count on retrieved nodes. Auto-reaffirm hot facts nearing expiry."""
    reaffirmed = 0
    if not node_ids:
        return 0
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE nodes SET access_count = access_count + 1 WHERE id = ANY(%s)", (node_ids,))
        cur.execute(
            """
            SELECT id FROM nodes
            WHERE id = ANY(%s)
              AND access_count >= %s
              AND valid_until IS NOT NULL
              AND valid_until BETWEEN NOW() AND NOW() + INTERVAL '48 hours'
            """,
            (node_ids, reaffirm_threshold),
        )
        hot = cur.fetchall()
        for row in hot:
            cur.execute("SELECT extend_validity(%s, 5)", (row["id"],))
            reaffirmed += 1
        conn.commit()
    if reaffirmed:
        log.info("Auto-reaffirmed %d hot facts", reaffirmed)
    return reaffirmed


def get_expiring_soon(project_id: uuid.UUID) -> list[dict]:
    """Find facts expiring within 24h for proactive refresh."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, content, summary, valid_until
            FROM nodes
            WHERE project_id = %s
              AND valid_until BETWEEN NOW() AND NOW() + INTERVAL '24 hours'
            """,
            (project_id,),
        )
        return cur.fetchall()


def resolve_conflicts(project_id: uuid.UUID) -> int:
    """Resolve conflicting answers in the same question cluster.

    Resolution order:
      1. Most recent valid_from (newer fact wins — temporal reasoning)
      2. Higher confidence (if same date)
      3. Loser gets expired + SUPERSEDES edge from winner
    """
    resolved = 0

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT cluster_id, array_agg(id ORDER BY confidence DESC) AS answer_ids
            FROM nodes
            WHERE label = 'ANSWER'
              AND project_id = %s
              AND valid_until > NOW()
              AND cluster_id IS NOT NULL
            GROUP BY cluster_id
            HAVING COUNT(*) > 1
            """,
            (project_id,),
        )
        clusters = cur.fetchall()

    for cluster in clusters:
        answer_ids = cluster["answer_ids"]

        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, confidence, provenance, valid_from FROM nodes WHERE id = ANY(%s)",
                (answer_ids,),
            )
            answers = cur.fetchall()

        # Sort by recency first (newer wins), then confidence
        answers.sort(
            key=lambda a: (a["valid_from"] or 0, a["confidence"] or 0),
            reverse=True,
        )

        winner = answers[0]
        for loser in answers[1:]:
            # Only expire if winner is strictly newer or higher confidence
            loser_time = loser["valid_from"]
            winner_time = winner["valid_from"]
            if (winner_time and loser_time and winner_time > loser_time) or \
               (winner["confidence"] or 0) > (loser["confidence"] or 0):
                with pool.connection() as conn, conn.cursor() as cur:
                    cur.execute(
                        "UPDATE nodes SET valid_until = NOW() WHERE id = %s AND valid_until > NOW()",
                        (loser["id"],),
                    )
                    conn.commit()
                try:
                    insert_edge(winner["id"], loser["id"], "SUPERSEDES")
                except Exception:
                    pass
                resolved += 1
                log.info("Resolved: %s (from %s) supersedes %s (from %s)",
                         winner["id"], winner_time, loser["id"], loser_time)

    return resolved
