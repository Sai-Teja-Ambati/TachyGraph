"""Specialist agents with focused tool sets."""
from __future__ import annotations

import logging

from strands import Agent
from tachyrag.agents.core import _get_model
from tachyrag.agents.tools import (
    memory_search, deep_search, memory_store, memory_observe,
    web_ingest, factchain, create_reminder, check_reminders, expiry_report,
)

log = logging.getLogger(__name__)


def _model(model_id: str | None = None):
    return _get_model(model_id)


def create_researcher(model_id: str | None = None) -> Agent:
    """Agent that finds NEW information — crawls, searches, stores."""
    return Agent(
        model=_model(model_id),
        system_prompt="""You are a Research Agent. Your job is to find information the user needs.

STRATEGY:
1. First search existing memory with memory_search
2. If results are sparse or insufficient, use web_ingest to crawl relevant URLs
3. After ingesting new content, search again to get the fresh results
4. Synthesize findings into a clear summary
5. Store the synthesis using memory_store for future reference

Always cite your sources. If you can't find enough information, say so.""",
        tools=[memory_search, deep_search, web_ingest, memory_store],
    )


def create_librarian(model_id: str | None = None) -> Agent:
    """Agent that RECALLS existing knowledge — searches, builds fact chains."""
    return Agent(
        model=_model(model_id),
        system_prompt="""You are a Librarian Agent. Your job is to recall and organize existing knowledge.

STRATEGY:
1. Search memory for the user's query
2. If the query is complex, use deep_search for multi-strategy retrieval
3. For verified information needs, use factchain for provenance-backed answers
4. If you find useful Q&A pairs, observe them to keep them fresh
5. Present findings organized by relevance and source

Always cite sources. Say "I don't have information about this" if memory is empty.""",
        tools=[memory_search, deep_search, factchain, memory_observe],
    )


def create_assistant(model_id: str | None = None) -> Agent:
    """Agent that performs ACTIONS — tasks, preferences, maintenance."""
    return Agent(
        model=_model(model_id),
        system_prompt="""You are an Assistant Agent. Your job is to help with tasks and memory management.

You can:
- Create reminders when the user asks to be reminded of something
- Check pending reminders
- Report on memory health (what's expiring)
- Store important information the user shares

Be proactive: if the user mentions a deadline, offer to create a reminder.""",
        tools=[create_reminder, check_reminders, expiry_report, memory_store, memory_observe],
    )
