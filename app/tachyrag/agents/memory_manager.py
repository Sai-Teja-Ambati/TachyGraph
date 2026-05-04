"""Memory management agent — autonomously maintains knowledge graph health."""
from __future__ import annotations

import logging

from strands import Agent, tool

from tachyrag.agents.core import _get_model
from tachyrag.agents.tools import memory_search, expiry_report, web_ingest

log = logging.getLogger(__name__)


@tool
def reaffirm_important(node_id: str, days: int = 10) -> str:
    """Extend the validity of an important fact that's about to expire."""
    import uuid
    from tachyrag.graph.temporal import reaffirm_fact
    reaffirm_fact(uuid.UUID(node_id), days)
    return f"Reaffirmed node {node_id} for {days} more days."


@tool
def compact_project_tool(project_id: str) -> str:
    """Remove near-duplicate nodes in a project to save space."""
    import uuid
    from tachyrag.graph.compaction import compact_project
    r = compact_project(uuid.UUID(project_id))
    return f"Compacted: {r['merged']} duplicates merged."


@tool
def list_projects() -> str:
    """List all projects with node counts."""
    from tachyrag.core.db import get_all_projects
    projects = get_all_projects()
    if not projects:
        return "No projects."
    return "\n".join(f"- {p['name']} (id: {p['id']}, nodes: {p.get('node_count', 0)})" for p in projects)


def create_memory_manager(model_id: str | None = None) -> Agent:
    return Agent(
        model=_get_model(model_id),
        system_prompt="""You are the Memory Manager Agent. Your job is to maintain the health of the knowledge graph.

TASKS:
1. Check the expiry report for facts about to expire
2. Reaffirm high-value facts (high access count) that are expiring
3. Let low-value facts expire naturally
4. Run compaction on projects with potential duplicates
5. Report what you did

Be conservative: only reaffirm facts that are clearly important (accessed frequently).
Let rarely-used facts expire to keep the graph lean.""",
        tools=[expiry_report, reaffirm_important, compact_project_tool, list_projects, memory_search],
    )


def run_memory_maintenance(model: str | None = None) -> dict:
    """Run the memory manager agent for autonomous graph maintenance."""
    agent = create_memory_manager(model)
    result = agent("Review the knowledge graph health. Check what's expiring, reaffirm important facts, compact duplicates, and report what you did.")
    return {"response": str(result), "mode": "memory_manager"}
