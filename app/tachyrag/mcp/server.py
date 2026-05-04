"""
TachyGraph MCP Server — Personal Memory for LLM Clients

Exposes the knowledge graph as MCP tools for Claude Desktop, VS Code, Cursor, etc.

Usage:
  python -m tachyrag.mcp.server

Config via env vars (same as FastAPI app):
  DATABASE_URL, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBED_MODEL
"""
from __future__ import annotations

import logging
import uuid

from mcp.server import FastMCP

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

mcp = FastMCP("tachygraph-memory")


def _uuid_or_none(val: str | None) -> uuid.UUID | None:
    if not val:
        return None
    try:
        return uuid.UUID(val)
    except ValueError:
        return None


@mcp.tool()
def memory_search(query: str, project_id: str = "", k: int = 10) -> str:
    """Search your personal knowledge graph for relevant facts, documents, and Q&A pairs."""
    from tachyrag.search.search import search
    results = search(query, project_id=_uuid_or_none(project_id), k=k)
    if not results:
        return "No relevant facts found in memory."
    facts = []
    for i, r in enumerate(results, 1):
        source = r.get("provenance", {}).get("source_url", "unknown") if r.get("provenance") else "unknown"
        facts.append(f"[{i}] {r.get('summary', r['content'][:200])} (source: {source})")
    return f"Found {len(results)} facts:\n" + "\n".join(facts)


@mcp.tool()
def memory_store(text: str, source: str = "mcp://memory_store", project_name: str = "default") -> str:
    """Store text as persistent knowledge in the graph (365-day expiry)."""
    from tachyrag.ingest.ingestor import ingest_document
    from tachyrag.core.db import ensure_project
    pid = uuid.uuid4()
    ensure_project(pid, project_name)
    result = ingest_document(text=text, source_url=source, project_id=pid, project_name=project_name)
    return f"Stored: {result.get('summaries', 0)} summaries in project '{project_name}' ({pid})"


@mcp.tool()
def memory_observe(question: str, answer: str, project_id: str = "") -> str:
    """Store a Q&A fact in the memory layer. Expires after 5 days unless reaffirmed."""
    from tachyrag.memory.clustering import find_or_create_question
    from tachyrag.memory.memory import add_answer
    from tachyrag.memory.weaver import weave_answer
    from tachyrag.core.db import ensure_project
    pid = _uuid_or_none(project_id) or uuid.uuid4()
    ensure_project(pid, "memory")
    q_id, is_new = find_or_create_question(question, pid)
    a_id = add_answer(question_id=q_id, answer_text=answer, confidence=0.90,
                      provenance={"source_url": "mcp://memory_observe"}, project_id=pid)
    if a_id:
        weave_answer(a_id)
        return f"Remembered: Q&A stored (question_id: {q_id}, new_question: {is_new})"
    return "Q&A not stored — confidence too low for eviction."


@mcp.tool()
def memory_recall(topic: str, k: int = 10) -> str:
    """Recall information about a topic. Searches across all projects."""
    from tachyrag.search.search import search
    results = search(topic, k=k)
    if not results:
        return f"I don't have any information about '{topic}' in memory."
    parts = [f"- {r.get('summary', r['content'][:300])}" for r in results]
    return f"Here's what I know about '{topic}':\n" + "\n".join(parts)


@mcp.tool()
def memory_chat(message: str, session_id: str = "") -> str:
    """Full RAG chat — searches memory for context, generates response, auto-learns."""
    from tachyrag.agents.core import run_agent
    result = run_agent(message, session_id=_uuid_or_none(session_id))
    return result["response"]


@mcp.tool()
def memory_tasks(action: str, description: str = "", due_days: int = 1, task_id: str = "") -> str:
    """Manage reminders: create, list_due, or complete tasks."""
    if action == "create":
        from tachyrag.graph.tasks import create_task
        t = create_task(description or "Reminder", due_days)
        return f"Task created: {t['description']} (due {t['due_at']})"
    elif action == "complete":
        from tachyrag.graph.tasks import complete_task
        tid = _uuid_or_none(task_id)
        if tid and complete_task(tid):
            return "Task completed."
        return "Task not found."
    else:
        from tachyrag.graph.tasks import get_due_tasks
        tasks = get_due_tasks()
        if not tasks:
            return "No tasks due."
        lines = [f"- {t['description']} (due {t['due_at']})" for t in tasks]
        return f"{len(tasks)} tasks due:\n" + "\n".join(lines)


@mcp.tool()
def memory_preferences(action: str = "get", preferences: str = "") -> str:
    """Get or set user preferences (response style, expertise level, etc.)."""
    if action == "set" and preferences:
        import json
        from tachyrag.graph.preferences import set_preferences
        try:
            prefs = json.loads(preferences)
        except (json.JSONDecodeError, TypeError):
            return "Invalid preferences JSON."
        r = set_preferences(prefs)
        return f"Preferences updated: {r['preferences']}"
    from tachyrag.graph.preferences import get_preferences
    prefs = get_preferences()
    return f"Current preferences: {prefs}"


@mcp.tool()
def memory_feedback(session_id: str, feedback: str, correction: str = "") -> str:
    """Correct a past answer: mark correct, wrong, or provide correction."""
    from tachyrag.chat.feedback import process_feedback
    sid = _uuid_or_none(session_id)
    if not sid:
        return "session_id required."
    result = process_feedback(sid, feedback, correction or None)
    return f"Feedback processed: {result['status']}"


@mcp.tool()
def memory_projects() -> str:
    """List all knowledge projects with summaries and node counts."""
    from tachyrag.core.db import get_all_projects
    projects = get_all_projects()
    if not projects:
        return "No projects in memory yet."
    lines = [f"- {p['name']} ({p.get('node_count', 0)} nodes) — {p.get('summary', 'No summary')}" for p in projects]
    return f"{len(projects)} projects:\n" + "\n".join(lines)


@mcp.tool()
def memory_status() -> str:
    """Health check: DB, Ollama, node count, expiring facts."""
    from tachyrag.core.db import pool
    from tachyrag.core.llm_client import check_health as llm_ok
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS total FROM nodes")
        total = cur.fetchone()["total"]
        cur.execute("SELECT COUNT(*) AS expiring FROM nodes WHERE valid_until BETWEEN NOW() AND NOW() + INTERVAL '24 hours'")
        expiring = cur.fetchone()["expiring"]
    return f"Status: DB=ok, LLM={'ok' if llm_ok() else 'down'}, Nodes={total}, Expiring(24h)={expiring}"


def main():
    log.info("Starting TachyGraph MCP server (stdio)...")
    mcp.run()


if __name__ == "__main__":
    main()
