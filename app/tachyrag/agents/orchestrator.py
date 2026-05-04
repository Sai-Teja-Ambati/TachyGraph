"""Multi-agent orchestrator: classifies intent, delegates to specialists."""
from __future__ import annotations

import logging
import uuid

from strands import Agent, tool

from tachyrag.config import LLM_PROVIDER
from tachyrag.agents.core import _get_model
from tachyrag.agents.specialists import create_researcher, create_librarian, create_assistant
from tachyrag.chat.session import create_session, add_message, get_history

log = logging.getLogger(__name__)

_specialists: dict[str, Agent] = {}


def _get_specialist(name: str, model_id: str | None = None) -> Agent:
    key = f"{name}:{LLM_PROVIDER}:{model_id or 'default'}"
    if key not in _specialists:
        if name == "researcher":
            _specialists[key] = create_researcher(model_id)
        elif name == "librarian":
            _specialists[key] = create_librarian(model_id)
        elif name == "assistant":
            _specialists[key] = create_assistant(model_id)
    return _specialists[key]


@tool
def delegate_to_researcher(query: str) -> str:
    """Delegate to the Research Agent for finding NEW information.
    Use when the user wants to learn about something, research a topic,
    or ingest content from a URL."""
    agent = _get_specialist("researcher")
    result = agent(query)
    return str(result)


@tool
def delegate_to_librarian(query: str) -> str:
    """Delegate to the Librarian Agent for RECALLING existing knowledge.
    Use when the user asks what they know, wants to find stored facts,
    or needs provenance-backed answers."""
    agent = _get_specialist("librarian")
    result = agent(query)
    return str(result)


@tool
def delegate_to_assistant(query: str) -> str:
    """Delegate to the Assistant Agent for ACTIONS like creating reminders,
    checking tasks, managing preferences, or storing information."""
    agent = _get_specialist("assistant")
    result = agent(query)
    return str(result)


def create_orchestrator(model_id: str | None = None) -> Agent:
    return Agent(
        model=_get_model(model_id),
        system_prompt="""You are the TachyGraph Orchestrator. You coordinate specialist agents to answer the user.

SPECIALISTS:
- delegate_to_researcher: For finding NEW information (web crawl, deep search, research topics)
- delegate_to_librarian: For RECALLING existing knowledge (search memory, fact chains, citations)
- delegate_to_assistant: For ACTIONS (reminders, tasks, storing info, memory health)

RULES:
- For simple greetings or follow-ups, respond directly without delegating
- For factual questions about stored knowledge, use the librarian
- For "research X" or "find out about X" or URLs, use the researcher
- For "remind me" or "save this" or "what's expiring", use the assistant
- For complex requests spanning multiple domains, call multiple specialists
- Synthesize specialist responses into a coherent answer for the user""",
        tools=[delegate_to_researcher, delegate_to_librarian, delegate_to_assistant],
    )


_orchestrator_cache: dict[str, Agent] = {}


def run_orchestrator(
    message: str,
    session_id: uuid.UUID | None = None,
    model: str | None = None,
) -> dict:
    """Run the multi-agent orchestrator."""
    key = f"{LLM_PROVIDER}:{model or 'default'}"
    if key not in _orchestrator_cache:
        _orchestrator_cache[key] = create_orchestrator(model)

    if not session_id:
        session_id = create_session()

    orchestrator = _orchestrator_cache[key]
    history = get_history(session_id, limit=10)
    messages = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in history] if history else None

    result = orchestrator(message, messages=messages)
    response_text = str(result)

    add_message(session_id, "user", message)
    add_message(session_id, "assistant", response_text)

    return {
        "response": response_text,
        "session_id": str(session_id),
        "mode": "orchestrator",
    }
