from __future__ import annotations

import re
from collections import Counter

# Minimal stop words to avoid bloating the JSONB with noise
_STOP = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once here there when where why how all each every both few "
    "more most other some such no nor not only own same so than too very s t d "
    "and but or if while that this it its he she they them their what which who".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


def compute_tf(text: str) -> tuple[dict[str, int], int]:
    """Return (term_frequency_map, doc_length) for BM25 storage."""
    tokens = tokenize(text)
    doc_length = len(tokens)
    tf = dict(Counter(tokens))
    return tf, doc_length
