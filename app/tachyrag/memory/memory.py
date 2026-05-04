from __future__ import annotations

import json
import logging
import uuid

import psycopg

from tachyrag.core.db import insert_edge, pool
from tachyrag.core.embedder import embed
from tachyrag.core.tfidf import compute_tf

log = logging.getLogger(__name__)


def add_answer(
    question_id: uuid.UUID,
    answer_text: str,
    confidence: float,
    provenance: dict,
    project_id: uuid.UUID,
) -> uuid.UUID | None:
    """Insert an ANSWER node and link to QUESTION. Returns answer id or None if rejected."""
    embedding = embed(answer_text)
    tf, dl = compute_tf(answer_text)

    with pool.connection() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                """INSERT INTO nodes (label, content, embedding, summary, tfidf, doc_length,
                   cluster_id, confidence, degree_cap, provenance, project_id)
                   VALUES ('ANSWER', %s, %s, %s, %s, %s, %s, %s, 10, %s, %s) RETURNING id""",
                (answer_text, embedding, answer_text[:200],
                 json.dumps(tf), dl, question_id, confidence,
                 json.dumps(provenance), project_id),
            )
            answer_id = cur.fetchone()["id"]
            cur.execute(
                "INSERT INTO edges (source_id, target_id, label) VALUES (%s, %s, 'ANSWERS')",
                (answer_id, question_id),
            )
            conn.commit()
            return answer_id
        except psycopg.errors.RaiseException as e:
            conn.rollback()
            log.info("Answer rejected by trigger: %s", e)
            return None
