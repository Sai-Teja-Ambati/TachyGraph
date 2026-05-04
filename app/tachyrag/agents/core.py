"""Core TachyGraph agent powered by Strands SDK. Supports Ollama, OpenAI, Anthropic."""
from __future__ import annotations

import logging
import uuid

from strands import Agent

from tachyrag.config import (
    LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
)
from tachyrag.agents.tools import ALL_TOOLS

log = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """You are TachyGraph, a personal knowledge agent. You have access to a knowledge graph containing the user's documents, Q&A pairs, and web content.

BEHAVIOR:
- Search your memory BEFORE answering factual questions about the user's data.
- If search returns nothing useful, say so honestly — do NOT hallucinate.
- If the user shares new information, store it using memory_store or memory_observe.
- If the user mentions a URL, offer to crawl and ingest it with web_ingest.
- If the user asks to be reminded of something, use create_reminder.
- Cite your sources when answering from memory (mention the source URL or fact number).
- For simple greetings or follow-up questions, just respond — don't search unnecessarily.
- If the first search doesn't find enough, try rephrasing or use deep_search.
- Be concise and direct."""

_agent_cache: dict[str, Agent] = {}


def _get_model(model_id: str | None = None):
    """Create a Strands model for the configured provider."""
    provider = LLM_PROVIDER

    if provider == "openai":
        from strands.models.openai import OpenAIModel
        return OpenAIModel(
            client_args={"api_key": OPENAI_API_KEY},
            model_id=model_id or OPENAI_MODEL,
        )
    elif provider == "anthropic":
        from strands.models.anthropic import AnthropicModel
        return AnthropicModel(
            client_args={"api_key": ANTHROPIC_API_KEY},
            model_id=model_id or ANTHROPIC_MODEL,
        )
    else:
        from strands.models.ollama import OllamaModel
        return OllamaModel(
            host=OLLAMA_BASE_URL,
            model_id=model_id or OLLAMA_MODEL,
        )


def get_agent(model_id: str | None = None) -> Agent:
    """Get or create a cached agent instance."""
    key = f"{LLM_PROVIDER}:{model_id or 'default'}"
    if key not in _agent_cache:
        model = _get_model(model_id)
        _agent_cache[key] = Agent(
            model=model,
            system_prompt=AGENT_SYSTEM_PROMPT,
            tools=ALL_TOOLS,
        )
        log.info("Created Strands agent: provider=%s, model=%s, tools=%d", LLM_PROVIDER, model_id or "default", len(ALL_TOOLS))
    return _agent_cache[key]


def run_agent(
    message: str,
    session_id: uuid.UUID | None = None,
    model: str | None = None,
) -> dict:
    """Run the agent on a user message. Returns response dict."""
    from tachyrag.chat.session import create_session, add_message, get_history

    if not session_id:
        session_id = create_session()

    agent = get_agent(model)

    history = get_history(session_id, limit=10)
    messages = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in history] if history else None

    result = agent(message, messages=messages)
    response_text = str(result)

    add_message(session_id, "user", message)
    add_message(session_id, "assistant", response_text)

    return {
        "response": response_text,
        "session_id": str(session_id),
    }
