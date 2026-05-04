from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

from tachyrag.core.db import pool

log = logging.getLogger(__name__)


def create_task(
    description: str,
    due_days: int = 1,
    project_id: uuid.UUID | None = None,
    related_node_id: uuid.UUID | None = None,
) -> dict:
    """Create a reminder/task. Returns task dict."""
    due_at = datetime.now() + timedelta(days=due_days)
    pid = project_id or uuid.uuid4()

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tasks (description, due_at, project_id, related_node_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id, description, due_at, completed, created_at
            """,
            (description, due_at, pid, related_node_id),
        )
        row = cur.fetchone()
        conn.commit()

    log.info("Task created: %s (due %s)", description[:80], due_at.isoformat())
    return {
        "id": str(row["id"]),
        "description": row["description"],
        "due_at": row["due_at"].isoformat(),
        "completed": row["completed"],
        "project_id": str(pid),
    }


def get_due_tasks(project_id: uuid.UUID | None = None) -> list[dict]:
    """Get tasks that are due (past or today)."""
    with pool.connection() as conn, conn.cursor() as cur:
        if project_id:
            cur.execute(
                "SELECT * FROM tasks WHERE project_id = %s AND NOT completed AND due_at <= NOW() + INTERVAL '24 hours' ORDER BY due_at",
                (project_id,),
            )
        else:
            cur.execute(
                "SELECT * FROM tasks WHERE NOT completed AND due_at <= NOW() + INTERVAL '24 hours' ORDER BY due_at"
            )
        rows = cur.fetchall()

    return [
        {
            "id": str(r["id"]),
            "description": r["description"],
            "due_at": r["due_at"].isoformat(),
            "project_id": str(r["project_id"]),
            "related_node_id": str(r["related_node_id"]) if r.get("related_node_id") else None,
        }
        for r in rows
    ]


def complete_task(task_id: uuid.UUID) -> bool:
    """Mark a task as completed."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE tasks SET completed = TRUE, completed_at = NOW() WHERE id = %s AND NOT completed RETURNING id",
            (task_id,),
        )
        row = cur.fetchone()
        conn.commit()
    return row is not None


def get_all_tasks(project_id: uuid.UUID | None = None, include_completed: bool = False) -> list[dict]:
    """List all tasks, optionally filtered by project and completion status."""
    with pool.connection() as conn, conn.cursor() as cur:
        conditions = []
        params: list = []
        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)
        if not include_completed:
            conditions.append("NOT completed")
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        cur.execute(f"SELECT * FROM tasks {where} ORDER BY due_at", params)
        rows = cur.fetchall()

    return [
        {
            "id": str(r["id"]),
            "description": r["description"],
            "due_at": r["due_at"].isoformat(),
            "completed": r["completed"],
            "project_id": str(r["project_id"]),
        }
        for r in rows
    ]
