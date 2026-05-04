"""Multi-provider LLM + embedding client.

Supports: Ollama, OpenAI, Anthropic (Claude), Google Gemini.
All providers use raw httpx — no SDK dependencies.

LLM_PROVIDER controls generation. EMBED_PROVIDER controls embeddings (defaults to LLM_PROVIDER).
This lets you use e.g. Claude for generation + OpenAI for embeddings.
"""
from __future__ import annotations

import json
import logging

import httpx

from tachyrag.config import (
    LLM_PROVIDER, EMBED_PROVIDER,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBED_MODEL,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_EMBED_MODEL,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_EMBED_MODEL,
)

log = logging.getLogger(__name__)

_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            timeout=120.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _client


def _embed_provider() -> str:
    return EMBED_PROVIDER or LLM_PROVIDER


# ===========================================================================
# Generate (blocking)
# ===========================================================================

def generate(prompt: str, model: str | None = None, max_tokens: int = 512) -> str:
    p = LLM_PROVIDER
    if p == "openai":
        return _openai_generate(prompt, model or OPENAI_MODEL, max_tokens)
    elif p == "anthropic":
        return _anthropic_generate(prompt, model or ANTHROPIC_MODEL, max_tokens)
    elif p == "gemini":
        return _gemini_generate(prompt, model or GEMINI_MODEL, max_tokens)
    return _ollama_generate(prompt, model or OLLAMA_MODEL, max_tokens)


def _ollama_generate(prompt: str, model: str, max_tokens: int) -> str:
    resp = _get_client().post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"num_predict": max_tokens}, "think": False},
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _openai_generate(prompt: str, model: str, max_tokens: int) -> str:
    resp = _get_client().post(
        f"{OPENAI_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": model, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _anthropic_generate(prompt: str, model: str, max_tokens: int) -> str:
    resp = _get_client().post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]},
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _gemini_generate(prompt: str, model: str, max_tokens: int) -> str:
    resp = _get_client().post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}",
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"maxOutputTokens": max_tokens}},
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


# ===========================================================================
# Generate stream (yields tokens)
# ===========================================================================

def generate_stream(prompt: str, model: str | None = None, max_tokens: int = 512):
    p = LLM_PROVIDER
    if p == "openai":
        yield from _openai_stream(prompt, model or OPENAI_MODEL, max_tokens)
    elif p == "anthropic":
        yield from _anthropic_stream(prompt, model or ANTHROPIC_MODEL, max_tokens)
    elif p == "gemini":
        yield _gemini_generate(prompt, model or GEMINI_MODEL, max_tokens)
    else:
        yield from _ollama_stream(prompt, model or OLLAMA_MODEL, max_tokens)


def _ollama_stream(prompt: str, model: str, max_tokens: int):
    with _get_client().stream(
        "POST", f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True,
              "options": {"num_predict": max_tokens}, "think": False},
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break
            except (json.JSONDecodeError, ValueError):
                continue


def _openai_stream(prompt: str, model: str, max_tokens: int):
    with _get_client().stream(
        "POST", f"{OPENAI_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": model, "max_tokens": max_tokens, "stream": True,
              "messages": [{"role": "user", "content": prompt}]},
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    yield token
            except (json.JSONDecodeError, ValueError, KeyError):
                continue


def _anthropic_stream(prompt: str, model: str, max_tokens: int):
    with _get_client().stream(
        "POST", "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": max_tokens, "stream": True,
              "messages": [{"role": "user", "content": prompt}]},
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
                if data.get("type") == "content_block_delta":
                    token = data.get("delta", {}).get("text", "")
                    if token:
                        yield token
            except (json.JSONDecodeError, ValueError):
                continue


# ===========================================================================
# Embeddings — routed by EMBED_PROVIDER (independent of LLM_PROVIDER)
# ===========================================================================

def embed_text(text: str, model: str | None = None) -> list[float]:
    p = _embed_provider()
    if p == "openai":
        return _openai_embed(text, model or OPENAI_EMBED_MODEL)
    elif p == "gemini":
        return _gemini_embed(text, model or GEMINI_EMBED_MODEL)
    return _ollama_embed(text, model or OLLAMA_EMBED_MODEL)


def embed_batch(texts: list[str], model: str | None = None) -> list[list[float]]:
    p = _embed_provider()
    if p == "openai":
        return _openai_embed_batch(texts, model or OPENAI_EMBED_MODEL)
    elif p == "gemini":
        return _gemini_embed_batch(texts, model or GEMINI_EMBED_MODEL)
    return _ollama_embed_batch(texts, model or OLLAMA_EMBED_MODEL)


# --- Ollama embeddings ---

def _ollama_embed(text: str, model: str) -> list[float]:
    resp = _get_client().post(f"{OLLAMA_BASE_URL}/api/embed", json={"model": model, "input": text})
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def _ollama_embed_batch(texts: list[str], model: str) -> list[list[float]]:
    resp = _get_client().post(f"{OLLAMA_BASE_URL}/api/embed", json={"model": model, "input": texts})
    resp.raise_for_status()
    return resp.json()["embeddings"]


# --- OpenAI embeddings ---

def _openai_embed(text: str, model: str) -> list[float]:
    resp = _get_client().post(
        f"{OPENAI_BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": model, "input": text},
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _openai_embed_batch(texts: list[str], model: str) -> list[list[float]]:
    resp = _get_client().post(
        f"{OPENAI_BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": model, "input": texts},
    )
    resp.raise_for_status()
    return [d["embedding"] for d in resp.json()["data"]]


# --- Gemini embeddings ---

def _gemini_embed(text: str, model: str) -> list[float]:
    resp = _get_client().post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={GEMINI_API_KEY}",
        json={"model": f"models/{model}", "content": {"parts": [{"text": text}]}},
    )
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def _gemini_embed_batch(texts: list[str], model: str) -> list[list[float]]:
    requests = [
        {"model": f"models/{model}", "content": {"parts": [{"text": t}]}}
        for t in texts
    ]
    resp = _get_client().post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents?key={GEMINI_API_KEY}",
        json={"requests": requests},
    )
    resp.raise_for_status()
    return [e["values"] for e in resp.json()["embeddings"]]


# ===========================================================================
# Utility
# ===========================================================================

def list_models() -> list[dict]:
    """List available models. Works for Ollama; returns empty for cloud providers."""
    try:
        resp = _get_client().get(f"{OLLAMA_BASE_URL}/api/tags")
        resp.raise_for_status()
        return resp.json().get("models", [])
    except Exception:
        return []


def check_health() -> bool:
    """Check if the LLM provider is reachable."""
    try:
        p = LLM_PROVIDER
        if p == "openai":
            resp = _get_client().get(f"{OPENAI_BASE_URL}/models",
                                     headers={"Authorization": f"Bearer {OPENAI_API_KEY}"})
            return resp.status_code == 200
        elif p == "anthropic":
            return bool(ANTHROPIC_API_KEY)
        elif p == "gemini":
            return bool(GEMINI_API_KEY)
        resp = _get_client().get(f"{OLLAMA_BASE_URL}/api/tags")
        return resp.status_code == 200
    except Exception:
        return False


def close():
    global _client
    if _client:
        _client.close()
        _client = None
