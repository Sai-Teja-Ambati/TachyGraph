"""TachyGraph tools for Strands agents. Each tool wraps an existing function."""
from __future__ import annotations

import uuid as _uuid_mod

from strands import tool


def _uuid(val: str | None) -> _uuid_mod.UUID | None:
    if not val:
        return None
    try:
        return _uuid_mod.UUID(val)
    except ValueError:
        return None


@tool
def memory_search(query: str, project_id: str = "", k: int = 10) -> str:
    """Search the knowledge graph for relevant facts and documents.
    Use this when the user asks a factual question or wants to recall something."""
    from tachyrag.search.search import search
    results = search(query, project_id=_uuid(project_id), k=k)
    if not results:
        return "No relevant facts found in memory."
    lines = []
    for i, r in enumerate(results, 1):
        source = r.get("provenance", {}).get("source_url", "unknown") if r.get("provenance") else "unknown"
        lines.append(f"[{i}] {r.get('summary', r['content'][:200])} (source: {source})")
    return "\n".join(lines)


@tool
def deep_search(query: str, project_id: str = "") -> str:
    """Multi-strategy search: disambiguates intent, runs parallel search strands
    (exact match, semantic, temporal, context-weave). Use for complex or ambiguous queries."""
    from tachyrag.search.search import deep_search as _deep
    result = _deep(query, project_id=_uuid(project_id))
    if not result["results"]:
        return "No results found with deep search."
    lines = []
    for r in result["results"]:
        strand = r.get("strand", "unknown")
        lines.append(f"[{strand}] {r.get('summary', r['content'][:200])}")
    return f"Intent: {result['intent'].get('intent', 'unknown')}\n" + "\n".join(lines)


@tool
def memory_store(text: str, source: str = "agent", project_name: str = "default") -> str:
    """Store text as persistent knowledge in the graph (365-day expiry).
    Use when the user shares important information worth remembering long-term."""
    from tachyrag.ingest.ingestor import ingest_document
    from tachyrag.core.db import ensure_project
    pid = _uuid_mod.uuid4()
    ensure_project(pid, project_name)
    result = ingest_document(text=text, source_url=source, project_id=pid, project_name=project_name)
    return f"Stored {result.get('summaries', 0)} summaries in project '{project_name}' ({pid})"


@tool
def memory_observe(question: str, answer: str, project_id: str = "") -> str:
    """Store a Q&A fact in the memory layer (5-day expiry, 10-slot eviction).
    Use for conversational facts that may change over time."""
    from tachyrag.memory.clustering import find_or_create_question
    from tachyrag.memory.memory import add_answer
    from tachyrag.memory.weaver import weave_answer
    from tachyrag.core.db import ensure_project
    pid = _uuid(project_id) or _uuid_mod.uuid4()
    ensure_project(pid, "memory")
    q_id, is_new = find_or_create_question(question, pid)
    a_id = add_answer(question_id=q_id, answer_text=answer, confidence=0.90,
                      provenance={"source_url": "agent://observe"}, project_id=pid)
    if a_id:
        weave_answer(a_id)
        return f"Remembered Q&A (question_id: {q_id}, new: {is_new})"
    return "Q&A not stored — confidence too low for eviction."


@tool
def web_ingest(url: str, limit: int = 20, depth: int = 2, project_name: str = "web") -> str:
    """Crawl a URL and ingest its content into the knowledge graph.
    Use when the user asks about a website or wants to add web content to memory."""
    from tachyrag.ingest.web_crawler import crawl_and_collect
    from tachyrag.ingest.ingestor import ingest_document
    from tachyrag.core.db import ensure_project
    pages = crawl_and_collect(url, limit=limit, depth=depth)
    if not pages:
        return f"No content found at {url}"
    pid = _uuid_mod.uuid4()
    ensure_project(pid, project_name)
    total = 0
    for page in pages:
        r = ingest_document(text=page["markdown"], source_url=page["url"], project_id=pid, project_name=project_name)
        total += r.get("summaries", 0)
    return f"Crawled {len(pages)} pages from {url}, created {total} summaries in project '{project_name}'"


@tool
def factchain(query: str, project_id: str = "") -> str:
    """Build a provenance-backed fact chain with citation trail.
    Use when the user needs verified, sourced information."""
    from tachyrag.search.responder import build_response, format_response
    pid = _uuid(project_id)
    if not pid:
        return "project_id required for factchain."
    chain = build_response(query, pid, k=5)
    return format_response(chain)


@tool
def create_reminder(description: str, due_days: int = 1) -> str:
    """Create a reminder or task. Use when the user says 'remind me',
    'don't forget', or asks to be notified about something."""
    from tachyrag.graph.tasks import create_task
    t = create_task(description, due_days)
    return f"Reminder set: '{description}' — due in {due_days} day(s)"


@tool
def check_reminders() -> str:
    """Check for pending reminders and tasks that are due within 24 hours."""
    from tachyrag.graph.tasks import get_due_tasks
    tasks = get_due_tasks()
    if not tasks:
        return "No reminders due."
    lines = [f"- {t['description']} (due {t['due_at']})" for t in tasks]
    return f"{len(tasks)} reminder(s) due:\n" + "\n".join(lines)


@tool
def expiry_report() -> str:
    """Check what knowledge is expiring soon. Use when the user asks about
    memory health or what's about to be forgotten."""
    from tachyrag.graph.compaction import get_expiry_report
    r = get_expiry_report()
    parts = [f"Total nodes: {r['total_nodes']}", f"Expiring 24h: {r['expiring_24h']}", f"Expiring 7d: {r['expiring_7d']}"]
    if r.get("hot_expiring"):
        parts.append("High-value facts expiring:")
        for h in r["hot_expiring"]:
            parts.append(f"  - {h['summary']} (accessed {h['access_count']}x)")
    return "\n".join(parts)


ALL_TOOLS = [
    memory_search, deep_search, memory_store, memory_observe,
    web_ingest, factchain, create_reminder, check_reminders, expiry_report,
]
