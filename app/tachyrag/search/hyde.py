"""HyDE: Hypothetical Document Embeddings.

Instead of embedding the raw query, generate a hypothetical answer and embed THAT.
The hypothetical answer uses document-like vocabulary, matching actual documents better.
"""
from __future__ import annotations

import logging

from tachyrag.core.llm_client import generate
from tachyrag.core.embedder import embed

log = logging.getLogger(__name__)

_HYDE_PROMPT = """/no_think
Write a short technical paragraph (50-100 words) that would answer this question.
Do NOT say "I don't know". Write as if you are a technical document, not a chatbot.
Output ONLY the paragraph, nothing else.

Question: {query}"""


def hyde_embed(query: str) -> list[float]:
    """Generate a hypothetical answer, embed it, return the embedding."""
    try:
        hypothetical = generate(_HYDE_PROMPT.format(query=query), max_tokens=200)
        return embed(hypothetical.strip())
    except Exception as e:
        log.debug("HyDE failed, falling back to direct embed: %s", e)
        return embed(query)
