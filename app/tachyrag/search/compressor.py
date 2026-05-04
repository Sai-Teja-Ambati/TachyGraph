"""Contextual compression: extract only relevant sentences from retrieved documents."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from tachyrag.core.llm_client import generate

log = logging.getLogger(__name__)

_COMPRESS_PROMPT = """/no_think
Extract ONLY the sentences from this text that help answer the question.
Output the relevant sentences verbatim. If nothing is relevant, output "NOT_RELEVANT".

Question: {query}

Text:
{text}"""

_MAX_UNCOMPRESSED = 600  # chars — don't compress short docs


def _compress_one(query: str, doc: dict) -> dict:
    content = doc.get("content", "")
    if len(content) <= _MAX_UNCOMPRESSED:
        return doc

    try:
        relevant = generate(
            _COMPRESS_PROMPT.format(query=query, text=content[:4000]),
            max_tokens=300,
        ).strip()
        if relevant and relevant != "NOT_RELEVANT":
            out = dict(doc)
            out["content"] = relevant
            out["compressed"] = True
            return out
    except Exception as e:
        log.debug("Compression failed for doc: %s", e)

    # Fallback: return summary + first 500 chars
    out = dict(doc)
    out["content"] = doc.get("summary", content[:500])
    return out


def compress_context(query: str, documents: list[dict], max_workers: int = 4) -> list[dict]:
    """Compress retrieved documents to only relevant sentences. Parallel."""
    if not documents:
        return []

    # Only compress docs that are long enough to benefit
    needs_compression = [d for d in documents if len(d.get("content", "")) > _MAX_UNCOMPRESSED]
    if not needs_compression:
        return documents

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_compress_one, query, d): i for i, d in enumerate(documents)}
        results = [None] * len(documents)
        for future in futures:
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = documents[idx]

    return [r for r in results if r is not None]
