from __future__ import annotations

import logging
import uuid

from tachyrag.core.db import pool

log = logging.getLogger(__name__)


def create_session(project_id: uuid.UUID | None = None) -> uuid.UUID:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (project_id) VALUES (%s) RETURNING id",
            (project_id,),
        )
        sid = cur.fetchone()["id"]
        conn.commit()
    return sid


def get_history(session_id: uuid.UUID, limit: int = 10) -> list[dict]:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE sessions SET last_active = NOW() WHERE id = %s",
            (session_id,),
        )
        cur.execute(
            """
            SELECT role, content, created_at
            FROM session_messages
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (session_id, limit * 2),
        )
        rows = cur.fetchall()
        conn.commit()
    rows.reverse()
    return [{"role": r["role"], "content": r["content"], "timestamp": str(r["created_at"])} for r in rows]


def add_message(session_id: uuid.UUID, role: str, content: str) -> uuid.UUID:
    with pool.connection() as conn, conn.cursor() as cur:
        # Ensure session exists
        cur.execute("SELECT id FROM sessions WHERE id = %s", (session_id,))
        if not cur.fetchone():
            cur.execute("INSERT INTO sessions (id) VALUES (%s)", (session_id,))
        cur.execute(
            "INSERT INTO session_messages (session_id, role, content) VALUES (%s, %s, %s) RETURNING id",
            (session_id, role, content),
        )
        msg_id = cur.fetchone()["id"]
        conn.commit()
    return msg_id


def list_sessions(limit: int = 20) -> list[dict]:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.id, s.project_id, s.created_at, s.last_active,
                   (SELECT COUNT(*) FROM session_messages m WHERE m.session_id = s.id) AS message_count
            FROM sessions s
            ORDER BY s.last_active DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [
            {
                "id": str(r["id"]),
                "project_id": str(r["project_id"]) if r["project_id"] else None,
                "created_at": str(r["created_at"]),
                "last_active": str(r["last_active"]),
                "message_count": r["message_count"],
            }
            for r in cur.fetchall()
        ]


def delete_session(session_id: uuid.UUID) -> bool:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM sessions WHERE id = %s RETURNING id", (session_id,))
        row = cur.fetchone()
        conn.commit()
    return row is not None


def get_last_assistant_message(session_id: uuid.UUID) -> dict | None:
    """Get the most recent assistant message in a session (for feedback)."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT content FROM session_messages WHERE session_id = %s AND role = 'assistant' ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        )
        row = cur.fetchone()
    return row if row else None


def get_last_user_message(session_id: uuid.UUID) -> dict | None:
    """Get the most recent user message in a session (for feedback)."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT content FROM session_messages WHERE session_id = %s AND role = 'user' ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        )
        row = cur.fetchone()
    return row if row else None
