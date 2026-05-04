from __future__ import annotations

import logging
import uuid

import psycopg

from tachyrag.config import WEAVE_TOP_K
from tachyrag.core.db import get_node, insert_edge, pool

log = logging.getLogger(__name__)


def weave_answer(answer_id: uuid.UUID) -> int:
    """Link answer to top-K similar answers in other clusters. Returns edges created."""
    node = get_node(answer_id)
    if node is None or node.get("embedding") is None:
        return 0

    embedding = node["embedding"]
    project_id = node["project_id"]
    cluster_id = node.get("cluster_id")

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM nodes
            WHERE label = 'ANSWER'
              AND project_id = %s
              AND id != %s
              AND (cluster_id IS NULL OR cluster_id != %s)
              AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (project_id, answer_id, cluster_id, embedding, WEAVE_TOP_K),
        )
        targets = cur.fetchall()

    created = 0
    for t in targets:
        try:
            insert_edge(answer_id, t["id"], "CONTEXT_OF")
            created += 1
        except (psycopg.errors.RaiseException, psycopg.errors.UniqueViolation):
            log.debug("Weave edge skipped for %s → %s", answer_id, t["id"])
    return created
