from __future__ import annotations

from functools import lru_cache

import numpy as np

from tachyrag.config import EMBEDDING_DIM
from tachyrag.core.llm_client import embed_text as _llm_embed, embed_batch as _llm_batch


def _normalize(v: list[float]) -> list[float]:
    a = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(a)
    if norm > 0:
        a = a / norm
    return a.tolist()


@lru_cache(maxsize=1024)
def embed(text: str) -> list[float]:
    raw = _llm_embed(text)
    vec = raw[:EMBEDDING_DIM] if len(raw) > EMBEDDING_DIM else raw
    return _normalize(vec)


def embed_batch(texts: list[str] | tuple[str, ...]) -> list[list[float]]:
    raw_batch = _llm_batch(list(texts))
    return [_normalize(r[:EMBEDDING_DIM] if len(r) > EMBEDDING_DIM else r) for r in raw_batch]


def embed_node(node: dict, use_summary: bool = True) -> list[float]:
    text = node.get("summary") if use_summary else None
    if not text:
        text = node["content"]
    return embed(text)
