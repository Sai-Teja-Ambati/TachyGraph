from __future__ import annotations

import logging
from collections import defaultdict

log = logging.getLogger(__name__)


def rerank_for_chat(
    results: list[dict],
    history: list[dict],
    k: int = 10,
    max_per_source: int = 3,
) -> list[dict]:
    """
    Rerank search results for chat context injection.
    1. Dedup: skip facts already cited in conversation history
    2. Diversity: max N results per source_url
    3. Keep original rank order within constraints
    """
    if not results:
        return []

    # Collect summaries already in history to avoid repetition
    history_text = " ".join(m.get("content", "") for m in history).lower()

    # Source diversity tracking
    source_counts: dict[str, int] = defaultdict(int)
    selected = []

    for r in results:
        summary = (r.get("summary") or "").lower()

        # Skip if this fact's summary is already substantially in the conversation
        if summary and len(summary) > 30 and summary[:60] in history_text:
            continue

        # Source diversity cap
        source = r.get("provenance", {}).get("source_url", "unknown") if r.get("provenance") else "unknown"
        if source_counts[source] >= max_per_source:
            continue
        source_counts[source] += 1

        selected.append(r)
        if len(selected) >= k:
            break

    return selected
