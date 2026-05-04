"""Query decomposition: break complex queries into focused sub-queries."""
from __future__ import annotations

import json
import logging

from tachyrag.core.llm_client import generate

log = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """/no_think
You are a query analyzer. Output ONLY valid JSON.

If this query contains multiple distinct information needs (comparison, multi-part, or compound question), decompose into 2-4 focused sub-queries.
If it's already focused on a single topic, return it as-is.

Query: "{query}"

Output: {{"sub_queries": ["query1", "query2"]}}"""


def decompose_query(query: str) -> list[str]:
    """Decompose a complex query into sub-queries. Returns [query] if already simple."""
    # Fast heuristic: skip decomposition for short/simple queries
    words = query.split()
    if len(words) < 6:
        return [query]

    has_complexity = any(w in query.lower() for w in [
        "compare", "vs", "versus", "difference", "between",
        "and also", "additionally", "as well as", "both",
    ])
    if not has_complexity:
        return [query]

    try:
        raw = generate(_DECOMPOSE_PROMPT.format(query=query), max_tokens=256)
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(raw[start:end + 1])
            subs = data.get("sub_queries", [query])
            if subs and all(isinstance(s, str) for s in subs):
                return subs[:4]
    except Exception as e:
        log.debug("Decomposition failed: %s", e)

    return [query]
