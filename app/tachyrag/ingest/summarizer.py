from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime

from tachyrag.config import TACHY_EXTRACT_PROMPT
from tachyrag.core.llm_client import generate

log = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    heading: str
    summary_text: str
    embedding: list[float]
    source_chunk_indices: list[int]
    body: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    temporal_context: str | None = None


def _parse_extraction(raw: str) -> dict | None:
    raw = raw.strip()
    # Strip <think>...</think> blocks from reasoning models
    think_end = raw.find("</think>")
    if think_end != -1:
        raw = raw[think_end + 8:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
        return None
    except (json.JSONDecodeError, ValueError):
        # Try to extract head via regex as last resort
        return None


def _extract_head_fallback(raw: str) -> str | None:
    """Regex fallback to extract head value from malformed JSON."""
    raw = raw.strip()
    think_end = raw.find("</think>")
    if think_end != -1:
        raw = raw[think_end + 8:].strip()
    match = re.search(r'"head"\s*:\s*"([^"]+)"', raw)
    if match:
        return match.group(1)
    return None


def _validate_date(date_str: str | None) -> str | None:
    """Normalize LLM-returned date to YYYY-MM-DD. The LLM decides what is a date."""
    if not date_str:
        return None
    s = str(date_str).strip()
    # Already YYYY-MM-DD
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except (ValueError, TypeError):
        pass
    # LLM returned YYYY-07-01 style or other valid formats — try common patterns
    for fmt in ("%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    for fmt in ("%b %Y", "%B %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-01")
        except (ValueError, TypeError):
            pass
    # LLM returned bare year as string — trust it since the prompt told it to only pick real dates
    if re.fullmatch(r"(19|20)\d{2}", s):
        return f"{s}-07-01"
    return None


def _safe_get(data: dict, key: str, default=None):
    """Get value from dict, handling keys that might have extra quotes."""
    if key in data:
        return data[key]
    quoted = f'"{key}"'
    if quoted in data:
        return data[quoted]
    return default


# Max chunks per summary group for plain text without headings
_GROUP_SIZE = 20


def generate_summary_for_chunk(content: str) -> str | None:
    """Generate a one-line summary for a single chunk using Ollama."""
    today = date.today().isoformat()
    try:
        prompt = TACHY_EXTRACT_PROMPT.format(system_date=today, text=content)
        raw = generate(prompt, max_tokens=1024)
        log.info("Ollama summary raw: %s", raw[:200])
        data = _parse_extraction(raw)
        if data:
            head = _safe_get(data, "head")
            if head:
                log.info("Extracted summary head: %s", head[:150])
                return head
        # Regex fallback
        head = _extract_head_fallback(raw)
        if head:
            log.info("Extracted summary head (regex fallback): %s", head[:150])
            return head
        return None
    except Exception as e:
        log.error("generate_summary_for_chunk failed: %s", e)
        return None


def _summarize_single(index: int, content: str, today: str) -> SummaryResult:
    """Summarize a single chunk — designed to run in a thread."""
    try:
        prompt = TACHY_EXTRACT_PROMPT.format(system_date=today, text=content)
        raw = generate(prompt, max_tokens=1024)
        data = _parse_extraction(raw)

        summary_text = None
        body = []
        keywords = []
        temporal = None

        if data:
            summary_text = _safe_get(data, "head")
            body = _safe_get(data, "body", [])
            keywords = _safe_get(data, "keywords", [])
            temporal = _validate_date(_safe_get(data, "temporal_context"))
            if isinstance(body, str):
                body = [body]
            if isinstance(keywords, str):
                keywords = [keywords]

        # Fallback: regex extract head from raw
        if not summary_text:
            summary_text = _extract_head_fallback(raw)

        # Final fallback
        if not summary_text:
            summary_text = f"Summary of chunk {index}"
            log.warning("No head extracted for chunk %d, using fallback", index)

    except Exception as e:
        log.error("Summarization failed for chunk %d: %s", index, e)
        summary_text = f"Summary of chunk {index}"
        body = []
        keywords = []
        temporal = None

    return SummaryResult(
        heading=f"chunk-{index}",
        summary_text=summary_text,
        embedding=[],
        source_chunk_indices=[index],
        body=body,
        keywords=keywords,
        temporal_context=temporal,
    )


def summarize_chunks(chunks: list[dict]) -> list[SummaryResult]:
    """One summary per chunk for fine-grained contextual retrieval."""
    today = date.today().isoformat()
    results: list[SummaryResult] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_summarize_single, i, c["content"], today): i
            for i, c in enumerate(chunks)
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                log.error("Summarize thread failed for chunk %d: %s", futures[future], e)

    return results
