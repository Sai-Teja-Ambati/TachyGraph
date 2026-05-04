from __future__ import annotations

import uuid

from tachyrag.config import QA_CLUSTER_GLOBAL, SIMILARITY_THRESHOLD
from tachyrag.core.db import insert_node, pool
from tachyrag.core.embedder import embed


def find_or_create_question(
    question_text: str,
    project_id: uuid.UUID,
) -> tuple[uuid.UUID, bool]:
    """Returns (question_node_id, is_new).

    When QA_CLUSTER_GLOBAL=true (default), searches ALL projects for an
    existing question hub with cosine >= 0.95.  This lets cross-project
    knowledge share the same Q&A hub.

    When QA_CLUSTER_GLOBAL=false, clustering is scoped to project_id.
    """
    question_embedding = embed(question_text)

    if QA_CLUSTER_GLOBAL:
        query = """
            SELECT id, project_id, 1 - (embedding <=> %s::vector) AS sim
            FROM nodes
            WHERE label = 'QUESTION'
                  AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 1
        """
        params = (question_embedding, question_embedding)
    else:
        query = """
            SELECT id, project_id, 1 - (embedding <=> %s::vector) AS sim
            FROM nodes
            WHERE label = 'QUESTION' AND project_id = %s
                  AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 1
        """
        params = (question_embedding, project_id, question_embedding)

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()

    if row and row["sim"] >= SIMILARITY_THRESHOLD:
        return row["id"], False

    new_id = insert_node(
        label="QUESTION",
        content=question_text,
        project_id=project_id,
        embedding=question_embedding,
        degree_cap=10,
    )
    return new_id, True
