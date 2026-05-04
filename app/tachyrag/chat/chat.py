from __future__ import annotations

import json
import logging
import uuid
from typing import Generator

from tachyrag.config import CHAT_CONTEXT_K, CHAT_HISTORY_TURNS, CHAT_SYSTEM_PROMPT, SEARCH_USE_HYDE, SEARCH_USE_COMPRESSION
from tachyrag.core.llm_client import generate, generate_stream
from tachyrag.search.search import search, estimate_k
from tachyrag.search.compressor import compress_context
from tachyrag.memory.clustering import find_or_create_question
from tachyrag.memory.memory import add_answer
from tachyrag.memory.weaver import weave_answer
from tachyrag.graph.temporal import reaffirm_fact
from tachyrag.graph.preferences import get_preference_prompt_context
from tachyrag.search.reranker import rerank_for_chat
from tachyrag.chat.session import get_history, add_message, create_session

log = logging.getLogger(__name__)


def _build_prompt(context: list[dict], history: list[dict], message: str, project_id: uuid.UUID | None = None) -> str:
    parts = [CHAT_SYSTEM_PROMPT]

    pref_ctx = get_preference_prompt_context(project_id)
    if pref_ctx:
        parts.append(f"USER PREFERENCES: {pref_ctx}")

    if context:
        facts = []
        for i, r in enumerate(context, 1):
            source = ""
            if r.get("provenance"):
                source = f" (source: {r['provenance'].get('source_url', 'unknown')})"
            facts.append(f"  [{i}] {r.get('summary', r['content'][:200])}{source}")
        parts.append("CONTEXT FROM MEMORY:\n" + "\n".join(facts))

    if history:
        turns = []
        for m in history:
            prefix = "User" if m["role"] == "user" else "Assistant"
            turns.append(f"  {prefix}: {m['content'][:500]}")
        parts.append("CONVERSATION HISTORY:\n" + "\n".join(turns))

    parts.append(f"User: {message}")
    return "\n\n---\n\n".join(parts)


def _post_chat(session_id: uuid.UUID, message: str, response: str, context: list[dict], project_id: uuid.UUID | None, auto_observe: bool) -> bool:
    """Post-chat: save history, auto-observe, reaffirm. Returns observed bool."""
    add_message(session_id, "user", message)
    add_message(session_id, "assistant", response)

    observed = False
    if auto_observe and len(message.split()) > 3 and len(response.split()) > 5:
        try:
            target_pid = project_id or (context[0]["project_id"] if context else None)
            if target_pid:
                q_id, _ = find_or_create_question(message, target_pid)
                a_id = add_answer(
                    question_id=q_id,
                    answer_text=response[:2000],
                    confidence=0.85,
                    provenance={"source_url": f"chat://session/{session_id}"},
                    project_id=target_pid,
                )
                if a_id:
                    weave_answer(a_id)
                    observed = True
        except Exception as e:
            log.debug("Auto-observe failed: %s", e)

    for r in context[:5]:
        try:
            reaffirm_fact(r["id"], extension_days=3)
        except Exception:
            pass

    return observed


def _build_sources(context: list[dict]) -> list[dict]:
    return [
        {
            "summary": r.get("summary", "")[:200],
            "source_url": r.get("provenance", {}).get("source_url", "unknown") if r.get("provenance") else "unknown",
            "project_id": str(r.get("project_id", "")),
            "rank": r.get("rank"),
        }
        for r in context
    ]


def chat(
    message: str,
    project_id: uuid.UUID | None = None,
    session_id: uuid.UUID | None = None,
    k: int = CHAT_CONTEXT_K,
    auto_observe: bool = True,
    model: str | None = None,
    bm25_weight: float | None = None,
    vector_weight: float | None = None,
    temporal_weight: float | None = None,
) -> dict:
    if not session_id:
        session_id = create_session(project_id)
    history = get_history(session_id, limit=CHAT_HISTORY_TURNS)

    # Adaptive K: skip search for greetings, adjust for complexity
    adaptive_k = estimate_k(message)
    effective_k = min(k, adaptive_k) if adaptive_k > 0 else 0

    if effective_k > 0:
        context = search(message, project_id=project_id, k=effective_k * 2, use_hyde=SEARCH_USE_HYDE,
                         bm25_weight=bm25_weight, vector_weight=vector_weight, temporal_weight=temporal_weight)
        context = rerank_for_chat(context, history, k=effective_k)
        if SEARCH_USE_COMPRESSION:
            context = compress_context(message, context)
    else:
        context = []

    prompt = _build_prompt(context, history, message, project_id)
    response = generate(prompt, model=model, max_tokens=2048)

    observed = _post_chat(session_id, message, response, context, project_id, auto_observe)

    return {
        "response": response,
        "session_id": str(session_id),
        "sources": _build_sources(context),
        "facts_retrieved": len(context),
        "observed": observed,
    }


def chat_stream(
    message: str,
    project_id: uuid.UUID | None = None,
    session_id: uuid.UUID | None = None,
    k: int = CHAT_CONTEXT_K,
    auto_observe: bool = True,
    model: str | None = None,
    bm25_weight: float | None = None,
    vector_weight: float | None = None,
    temporal_weight: float | None = None,
) -> Generator[str, None, None]:
    """Yield SSE events as tokens arrive. Final event includes metadata."""
    if not session_id:
        session_id = create_session(project_id)
    history = get_history(session_id, limit=CHAT_HISTORY_TURNS)

    adaptive_k = estimate_k(message)
    effective_k = min(k, adaptive_k) if adaptive_k > 0 else 0

    if effective_k > 0:
        context = search(message, project_id=project_id, k=effective_k * 2, use_hyde=SEARCH_USE_HYDE,
                         bm25_weight=bm25_weight, vector_weight=vector_weight, temporal_weight=temporal_weight)
        context = rerank_for_chat(context, history, k=effective_k)
        if SEARCH_USE_COMPRESSION:
            context = compress_context(message, context)
    else:
        context = []

    prompt = _build_prompt(context, history, message, project_id)

    full_response = []
    for token in generate_stream(prompt, model=model, max_tokens=2048):
        full_response.append(token)
        yield f"data: {json.dumps({'token': token})}\n\n"

    response_text = "".join(full_response)
    observed = _post_chat(session_id, message, response_text, context, project_id, auto_observe)

    yield f"data: {json.dumps({'done': True, 'session_id': str(session_id), 'sources': _build_sources(context), 'facts_retrieved': len(context), 'observed': observed})}\n\n"
