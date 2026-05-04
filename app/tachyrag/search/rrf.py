"""Reciprocal Rank Fusion: merge ranked lists by position, not score."""
from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_n: int = 21,
) -> list[dict]:
    """Merge multiple ranked result lists using RRF.

    RRF_score(d) = Σ 1/(k + rank_i(d))

    k=60 is the standard constant from the original RRF paper.
    Each signal contributes equally by rank position regardless of score scale.
    """
    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            doc_id = str(doc["id"])
            scores[doc_id] += 1.0 / (k + rank + 1)
            if doc_id not in docs:
                docs[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda did: scores[did], reverse=True)
    results = []
    for did in sorted_ids[:top_n]:
        doc = docs[did]
        doc["rrf_score"] = round(scores[did], 6)
        results.append(doc)
    return results
