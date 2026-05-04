from __future__ import annotations

import logging

from tachyrag.core.db import pool

log = logging.getLogger(__name__)


def run_maintenance() -> dict:
    """Run all periodic maintenance tasks. Called by APScheduler cron."""
    stats = {}

    # 1. Hard-delete expired nodes
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM nodes WHERE valid_until IS NOT NULL AND valid_until < NOW() RETURNING id")
        stats["expired_purged"] = cur.rowcount
        conn.commit()

    # 2. Compact all projects
    from tachyrag.graph.compaction import compact_project
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM nodes WHERE label = 'PROJECT'")
        projects = [r["id"] for r in cur.fetchall()]
    total_merged = 0
    for pid in projects:
        try:
            r = compact_project(pid)
            total_merged += r.get("merged", 0)
        except Exception as e:
            log.warning("Compaction failed for %s: %s", pid, e)
    stats["compacted"] = total_merged

    # 3. Clean stale sessions (> 30 days)
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM sessions WHERE last_active < NOW() - INTERVAL '30 days' RETURNING id")
        stats["sessions_cleaned"] = cur.rowcount
        conn.commit()

    # 4. Clean completed tasks (> 7 days)
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM tasks WHERE completed AND completed_at < NOW() - INTERVAL '7 days' RETURNING id")
        stats["tasks_cleaned"] = cur.rowcount
        conn.commit()

    # 5. Stats
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS total FROM nodes")
        stats["total_nodes"] = cur.fetchone()["total"]

    log.info("Maintenance complete: %s", stats)
    return stats
