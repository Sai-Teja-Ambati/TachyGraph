"""Standalone research agent — autonomously researches a topic."""
from __future__ import annotations

import logging
import uuid

from tachyrag.agents.specialists import create_researcher
from tachyrag.chat.session import create_session, add_message

log = logging.getLogger(__name__)


def research(
    topic: str,
    session_id: uuid.UUID | None = None,
    model: str | None = None,
) -> dict:
    """Run the research agent on a topic. It will search, crawl, ingest, and synthesize."""
    if not session_id:
        session_id = create_session()

    agent = create_researcher(model)
    prompt = f"Research the following topic thoroughly. Search existing memory first, then crawl relevant web sources if needed. Synthesize your findings and store the key points.\n\nTopic: {topic}"

    result = agent(prompt)
    response_text = str(result)

    add_message(session_id, "user", f"[Research] {topic}")
    add_message(session_id, "assistant", response_text)

    return {
        "response": response_text,
        "session_id": str(session_id),
        "mode": "research",
    }
