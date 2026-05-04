from __future__ import annotations

import json
import logging

from tachyrag.core.llm_client import generate

log = logging.getLogger(__name__)

_PROMPT = """/no_think
You are a JSON extraction agent. Output ONLY a single valid JSON object. No markdown, no explanation, no preamble.

Given the user query and context, extract structured search intent.

Context: {context}
User Query: "{query}"

Required JSON format:
{{
    "intent": "debugging|reference|comparison|history",
    "rephrased": "specific rephrased query",
    "entities": ["extracted", "technical", "terms"],
    "temporal_scope": "recent|all_time|specific_date",
    "confidence": 0.0-1.0
}}"""


class OllamaDisambiguator:
    def disambiguate(self, raw_query: str, context: dict | None = None) -> dict:
        prompt = _PROMPT.format(
            context=context.get("recent_topics", []) if context else [],
            query=raw_query,
        )
        try:
            raw = generate(prompt, max_tokens=256)
            raw = raw.strip()
            think_end = raw.find("</think>")
            if think_end != -1:
                raw = raw[think_end + 8:].strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start : end + 1])
        except Exception as e:
            log.warning("Disambiguation failed, using passthrough: %s", e)

        return {
            "intent": "reference",
            "rephrased": raw_query,
            "entities": [],
            "temporal_scope": "all_time",
            "confidence": 0.5,
        }
