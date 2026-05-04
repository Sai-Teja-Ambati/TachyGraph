from __future__ import annotations

import json
from dataclasses import dataclass

from tachyrag.config import CONFIDENCE_FLOOR, QA_SKIP_LLM_EXTRACT
from tachyrag.core.llm_client import generate

_EXTRACT_PROMPT = """/no_think
You are a JSON extraction agent. Output ONLY a single valid JSON object. No markdown, no explanation, no preamble.

Extract a Q&A fact from this interaction. If no explicit factual Q&A is present, return {{"skip": true}}.

Required JSON format:
{{"question": "...", "answer": "...", "subject": "...", "predicate": "...", "object": "...", "confidence": 0.0-1.0}}

Interaction:
{text}
"""


@dataclass
class ObservedFact:
    question: str
    answer: str
    subject: str
    predicate: str
    object: str
    confidence: float


def _strip_think(raw: str) -> str:
    """Strip <think>...</think> blocks from reasoning models."""
    raw = raw.strip()
    think_end = raw.find("</think>")
    if think_end != -1:
        raw = raw[think_end + 8:].strip()
    return raw


def _parse_json(raw: str) -> dict | None:
    raw = _strip_think(raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def _fast_parse(text: str) -> ObservedFact | None:
    """Parse Q: ... A: ... directly without LLM. Returns None if format doesn't match."""
    import re
    m = re.search(r'Q:\s*(.+?)\s*A:\s*(.+)', text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    question = m.group(1).strip().rstrip('?') + '?'
    answer = m.group(2).strip()
    if not question or not answer or len(answer) < 5:
        return None
    return ObservedFact(
        question=question,
        answer=answer,
        subject=question.split()[0] if question else "",
        predicate="is",
        object=answer[:50],
        confidence=0.90,
    )


def observe(interaction_text: str) -> ObservedFact | None:
    if QA_SKIP_LLM_EXTRACT:
        fast = _fast_parse(interaction_text)
        if fast:
            return fast

    raw = generate(_EXTRACT_PROMPT.format(text=interaction_text[:1500]), max_tokens=256)

    data = _parse_json(raw)
    if not data:
        return None

    if data.get("skip"):
        return None

    confidence = float(data.get("confidence", 0))
    if confidence < CONFIDENCE_FLOOR:
        return None

    required = ("question", "answer", "subject", "predicate", "object")
    if not all(data.get(k) for k in required):
        return None

    return ObservedFact(
        question=data["question"],
        answer=data["answer"],
        subject=data["subject"],
        predicate=data["predicate"],
        object=data["object"],
        confidence=confidence,
    )
