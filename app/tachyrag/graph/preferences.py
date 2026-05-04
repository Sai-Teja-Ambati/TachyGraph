from __future__ import annotations

import json
import logging
import uuid

from tachyrag.core.db import pool

log = logging.getLogger(__name__)

_DEFAULT_PREFS = {
    "response_style": "concise",       # concise, detailed, technical
    "expertise_level": "intermediate",  # beginner, intermediate, expert
    "language": "english",
    "topics_of_interest": [],
    "max_response_length": 500,
}


def get_preferences(project_id: uuid.UUID | None = None) -> dict:
    """Get user preferences. Falls back to defaults for missing keys."""
    with pool.connection() as conn, conn.cursor() as cur:
        if project_id:
            cur.execute(
                "SELECT content FROM nodes WHERE label = 'PREFERENCE' AND project_id = %s ORDER BY created_at DESC LIMIT 1",
                (project_id,),
            )
        else:
            cur.execute("SELECT content FROM nodes WHERE label = 'PREFERENCE' ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()

    prefs = dict(_DEFAULT_PREFS)
    if row:
        try:
            stored = json.loads(row["content"])
            prefs.update(stored)
        except (json.JSONDecodeError, TypeError):
            pass
    return prefs


def set_preferences(prefs: dict, project_id: uuid.UUID | None = None) -> dict:
    """Set or update user preferences. Merges with existing."""
    current = get_preferences(project_id)
    current.update(prefs)
    content = json.dumps(current)
    pid = project_id or uuid.uuid4()

    with pool.connection() as conn, conn.cursor() as cur:
        # Upsert: expire old preference, insert new
        if project_id:
            cur.execute(
                "UPDATE nodes SET valid_until = NOW() WHERE label = 'PREFERENCE' AND project_id = %s AND (valid_until IS NULL OR valid_until > NOW())",
                (project_id,),
            )
        cur.execute(
            "INSERT INTO nodes (label, content, degree_cap, project_id) VALUES ('PREFERENCE', %s, 0, %s) RETURNING id",
            (content, pid),
        )
        node_id = cur.fetchone()["id"]
        conn.commit()

    log.info("Preferences updated: %s", content[:200])
    return {"id": str(node_id), "project_id": str(pid), "preferences": current}


def get_preference_prompt_context(project_id: uuid.UUID | None = None) -> str:
    """Build a prompt fragment from preferences for chat context injection."""
    prefs = get_preferences(project_id)
    parts = []
    if prefs.get("response_style") != "concise":
        parts.append(f"Response style: {prefs['response_style']}")
    if prefs.get("expertise_level") != "intermediate":
        parts.append(f"User expertise: {prefs['expertise_level']}")
    if prefs.get("topics_of_interest"):
        parts.append(f"Topics of interest: {', '.join(prefs['topics_of_interest'])}")
    return "; ".join(parts) if parts else ""
