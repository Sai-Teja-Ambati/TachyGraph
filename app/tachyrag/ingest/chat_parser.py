from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class QAPair:
    question: str
    answer: str
    conversation_title: str
    source: str  # chatgpt, claude, gemini, raw


# ---------------------------------------------------------------------------
# ChatGPT — conversations.json export
# ---------------------------------------------------------------------------

def _parse_chatgpt_export(data: list[dict]) -> list[QAPair]:
    """Parse ChatGPT Settings → Data Controls → Export Data → conversations.json"""
    pairs = []
    for conv in data:
        title = conv.get("title", "Untitled")
        mapping = conv.get("mapping", {})

        # Build ordered message list from the mapping tree
        messages = []
        for node in mapping.values():
            msg = node.get("message")
            if not msg or not msg.get("content"):
                continue
            role = msg.get("author", {}).get("role", "")
            parts = msg["content"].get("parts", [])
            text = "\n".join(str(p) for p in parts if isinstance(p, str)).strip()
            if text and role in ("user", "assistant"):
                messages.append((role, text, msg.get("create_time", 0)))

        messages.sort(key=lambda m: m[2] or 0)

        # Pair consecutive user→assistant messages
        i = 0
        while i < len(messages) - 1:
            if messages[i][0] == "user" and messages[i + 1][0] == "assistant":
                pairs.append(QAPair(
                    question=messages[i][1],
                    answer=messages[i + 1][1],
                    conversation_title=title,
                    source="chatgpt",
                ))
                i += 2
            else:
                i += 1
    return pairs


def _parse_chatgpt_single(data: dict) -> list[QAPair]:
    """Parse a single ChatGPT conversation object (same structure as export, but one conv)."""
    return _parse_chatgpt_export([data])


# ---------------------------------------------------------------------------
# Claude — JSON export
# ---------------------------------------------------------------------------

def _parse_claude_export(data: list[dict]) -> list[QAPair]:
    """Parse Claude export JSON — array of conversations with chat_messages."""
    pairs = []
    for conv in data:
        title = conv.get("name", conv.get("title", "Untitled"))
        messages = conv.get("chat_messages", [])

        i = 0
        while i < len(messages) - 1:
            curr = messages[i]
            nxt = messages[i + 1]
            q_role = curr.get("sender", curr.get("role", ""))
            a_role = nxt.get("sender", nxt.get("role", ""))
            q_text = curr.get("text", curr.get("content", "")).strip()
            a_text = nxt.get("text", nxt.get("content", "")).strip()

            if q_role in ("human", "user") and a_role in ("assistant",) and q_text and a_text:
                pairs.append(QAPair(question=q_text, answer=a_text, conversation_title=title, source="claude"))
                i += 2
            else:
                i += 1
    return pairs


def _parse_claude_single(data: dict) -> list[QAPair]:
    """Parse a single Claude conversation object."""
    return _parse_claude_export([data])


# ---------------------------------------------------------------------------
# Gemini — Google Takeout JSON
# ---------------------------------------------------------------------------

def _parse_gemini_export(data: list[dict]) -> list[QAPair]:
    """Parse Gemini Takeout — array of conversations with messages."""
    pairs = []
    for conv in data:
        title = conv.get("title", "Untitled")
        messages = conv.get("messages", [])

        i = 0
        while i < len(messages) - 1:
            curr = messages[i]
            nxt = messages[i + 1]
            q_role = curr.get("role", "")
            a_role = nxt.get("role", "")
            q_text = curr.get("text", curr.get("content", "")).strip()
            a_text = nxt.get("text", nxt.get("content", "")).strip()

            if q_role == "user" and a_role == "model" and q_text and a_text:
                pairs.append(QAPair(question=q_text, answer=a_text, conversation_title=title, source="gemini"))
                i += 2
            else:
                i += 1
    return pairs


def _parse_gemini_single(data: dict) -> list[QAPair]:
    """Parse a single Gemini conversation object."""
    return _parse_gemini_export([data])


# ---------------------------------------------------------------------------
# Raw text — pasted conversation (auto-detect format)
# ---------------------------------------------------------------------------

_ROLE_PATTERNS = [
    # ChatGPT UI copy-paste
    (re.compile(r"^(?:You|User|Human)\s*:", re.IGNORECASE), "user"),
    (re.compile(r"^(?:ChatGPT|Assistant|Claude|Gemini|AI|Model)\s*:", re.IGNORECASE), "assistant"),
    # Q: / A: format
    (re.compile(r"^Q\s*:", re.IGNORECASE), "user"),
    (re.compile(r"^A\s*:", re.IGNORECASE), "assistant"),
]


def _parse_raw_text(text: str) -> list[QAPair]:
    """Parse raw pasted conversation text by detecting role prefixes."""
    lines = text.strip().split("\n")
    messages: list[tuple[str, list[str]]] = []
    current_role = None
    current_lines: list[str] = []

    for line in lines:
        matched = False
        for pattern, role in _ROLE_PATTERNS:
            m = pattern.match(line)
            if m:
                if current_role and current_lines:
                    messages.append((current_role, current_lines))
                current_role = role
                current_lines = [line[m.end():].strip()]
                matched = True
                break
        if not matched and current_role:
            current_lines.append(line)

    if current_role and current_lines:
        messages.append((current_role, current_lines))

    pairs = []
    i = 0
    while i < len(messages) - 1:
        if messages[i][0] == "user" and messages[i + 1][0] == "assistant":
            q = "\n".join(messages[i][1]).strip()
            a = "\n".join(messages[i + 1][1]).strip()
            if q and a:
                pairs.append(QAPair(question=q, answer=a, conversation_title="Pasted Conversation", source="raw"))
            i += 2
        else:
            i += 1
    return pairs


# ---------------------------------------------------------------------------
# Auto-detect and parse
# ---------------------------------------------------------------------------

def _detect_and_parse_json(data) -> list[QAPair]:
    """Auto-detect JSON format and parse accordingly."""
    # Array of conversations
    if isinstance(data, list) and data:
        sample = data[0]
        if "mapping" in sample:
            return _parse_chatgpt_export(data)
        if "chat_messages" in sample:
            return _parse_claude_export(data)
        if "messages" in sample:
            return _parse_gemini_export(data)

    # Single conversation object
    if isinstance(data, dict):
        if "mapping" in data:
            return _parse_chatgpt_single(data)
        if "chat_messages" in data:
            return _parse_claude_single(data)
        if "messages" in data:
            return _parse_gemini_single(data)

    return []


def parse_chat_input(text: str | None = None, json_data: dict | list | None = None) -> list[QAPair]:
    """
    Unified entry point. Accepts either:
      - json_data: parsed JSON (ChatGPT/Claude/Gemini export)
      - text: raw pasted conversation text

    Returns list of QAPair.
    """
    if json_data is not None:
        pairs = _detect_and_parse_json(json_data)
        if pairs:
            return pairs

    if text:
        # Try parsing as JSON first
        try:
            data = json.loads(text)
            pairs = _detect_and_parse_json(data)
            if pairs:
                return pairs
        except (json.JSONDecodeError, ValueError):
            pass
        # Fall back to raw text parsing
        return _parse_raw_text(text)

    return []
