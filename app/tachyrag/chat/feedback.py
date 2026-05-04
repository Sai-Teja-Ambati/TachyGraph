from __future__ import annotations

import logging
import uuid

from tachyrag.core.db import pool
from tachyrag.memory.clustering import find_or_create_question
from tachyrag.memory.memory import add_answer
from tachyrag.memory.weaver import weave_answer
from tachyrag.chat.session import get_last_assistant_message, get_last_user_message

log = logging.getLogger(__name__)


def process_feedback(
    session_id: uuid.UUID,
    feedback: str,
    correction: str | None = None,
) -> dict:
    """
    Process user feedback on the last chat response.
      "correct"    → reaffirm the auto-observed Q&A (confidence → 0.95)
      "wrong"      → expire the auto-observed answer
      "correction" → insert new answer with confidence=0.98, supersedes old
    """
    user_msg = get_last_user_message(session_id)
    asst_msg = get_last_assistant_message(session_id)

    if not user_msg or not asst_msg:
        return {"status": "error", "reason": "No recent Q&A found in session"}

    question = user_msg["content"]
    old_answer = asst_msg["content"]

    # Find the question hub and the auto-observed answer
    with pool.connection() as conn, conn.cursor() as cur:
        # Find the answer node created by auto-observe for this session
        cur.execute(
            """
            SELECT n.id, n.cluster_id, n.project_id
            FROM nodes n
            WHERE n.label = 'ANSWER'
              AND n.provenance->>'source_url' LIKE %s
              AND (n.valid_until IS NULL OR n.valid_until > NOW())
            ORDER BY n.created_at DESC LIMIT 1
            """,
            (f"chat://session/{session_id}%",),
        )
        answer_row = cur.fetchone()

    if feedback == "correct" and answer_row:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute("UPDATE nodes SET confidence = 0.95 WHERE id = %s", (answer_row["id"],))
            cur.execute("SELECT extend_validity(%s, 10)", (answer_row["id"],))
            conn.commit()
        return {"status": "reaffirmed", "answer_id": str(answer_row["id"]), "confidence": 0.95}

    elif feedback == "wrong" and answer_row:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute("UPDATE nodes SET valid_until = NOW(), confidence = 0.0 WHERE id = %s", (answer_row["id"],))
            conn.commit()
        return {"status": "expired", "answer_id": str(answer_row["id"])}

    elif feedback == "correction" and correction:
        project_id = answer_row["project_id"] if answer_row else None
        if not project_id:
            return {"status": "error", "reason": "No project context for correction"}

        # Expire old answer
        if answer_row:
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute("UPDATE nodes SET valid_until = NOW() WHERE id = %s", (answer_row["id"],))
                conn.commit()

        # Insert correction with high confidence
        q_id, _ = find_or_create_question(question, project_id)
        a_id = add_answer(
            question_id=q_id,
            answer_text=correction,
            confidence=0.98,
            provenance={"source_url": f"chat://feedback/{session_id}"},
            project_id=project_id,
        )
        if a_id:
            weave_answer(a_id)
            return {"status": "corrected", "old_answer_id": str(answer_row["id"]) if answer_row else None, "new_answer_id": str(a_id), "confidence": 0.98}
        return {"status": "rejected", "reason": "Correction confidence too low for eviction"}

    return {"status": "no_action", "reason": f"Unknown feedback type or missing data: {feedback}"}
