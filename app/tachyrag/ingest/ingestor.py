from __future__ import annotations

import logging
import re
import uuid

import psycopg

from tachyrag.ingest.chunker import Chunk, chunk_document
from tachyrag.config import MMR_ON_INGEST
from tachyrag.core.db import bulk_insert_summaries, ensure_project, insert_edge, pool, update_project_summary
from tachyrag.core.embedder import embed_batch
from tachyrag.graph.mmr import mmr_link_summary
from tachyrag.core.llm_client import generate
from tachyrag.ingest.summarizer import SummaryResult, summarize_chunks
from tachyrag.core.tfidf import compute_tf

log = logging.getLogger(__name__)


def _sanitize_text(text: str) -> str:
    text = text.replace("\\", " ")
    text = text.replace("\x00", "")
    text = re.sub(r'["\u201c\u201d\u201e\u201f]', '"', text)
    text = re.sub(r"['\u2018\u2019\u201a\u201b]", "'", text)
    text = re.sub(r"[^\x20-\x7E\n\r\t]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _safe_insert_edge(source_id: uuid.UUID, target_id: uuid.UUID, label: str) -> bool:
    try:
        insert_edge(source_id, target_id, label)
        return True
    except (psycopg.errors.RaiseException, psycopg.errors.UniqueViolation):
        return False


def _generate_project_summary(project_id: uuid.UUID, chunks: list[Chunk]) -> None:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT summary FROM nodes WHERE id = %s AND label = 'PROJECT'", (project_id,))
        row = cur.fetchone()
        if row and row["summary"]:
            return
    try:
        sample = " ".join(_sanitize_text(c.content)[:300] for c in chunks[:5])
        prompt = f"Summarize the following content in exactly 30 words or less. Output only the summary, nothing else.\n\n{sample[:2000]}"
        summary = generate(prompt, max_tokens=60).strip()
        update_project_summary(project_id, summary)
        log.info("Project summary generated: %s", summary[:80])
    except Exception as e:
        log.warning("Failed to generate project summary: %s", e)


def _check_duplicate_source(source_url: str, project_id: uuid.UUID) -> bool:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM nodes WHERE project_id = %s AND label = 'SUMMARY' AND provenance->>'source_url' = %s LIMIT 1",
            (project_id, source_url),
        )
        return cur.fetchone() is not None


def ingest_document(
    text: str,
    source_url: str,
    project_id: uuid.UUID,
    project_name: str = "default",
    summarize_chunks_inline: bool = False,
) -> dict:
    ensure_project(project_id, project_name)

    if _check_duplicate_source(source_url, project_id):
        log.info("Source already ingested: %s", source_url)
        return {"summaries": 0, "project_id": str(project_id), "status": "duplicate"}

    text = _sanitize_text(text)
    chunks: list[Chunk] = chunk_document(text, source_url)
    log.info("Document chunked into %d pages", len(chunks))

    # Phase 1: Generate summaries via LLM (parallel, no embedding yet)
    chunk_dicts = [{"content": _sanitize_text(c.content), "provenance": c.provenance} for c in chunks]
    summaries: list[SummaryResult] = summarize_chunks(chunk_dicts)

    # Phase 2: Batch embed all summary texts in one Ollama call
    summary_texts = [s.summary_text for s in summaries]
    embeddings = embed_batch(tuple(summary_texts))
    for s, emb in zip(summaries, embeddings):
        s.embedding = emb
    log.info("Batch embedded %d summaries", len(embeddings))

    # Phase 3: Build rows — TF computed from FULL CHUNK CONTENT (not keywords)
    rows = []
    for i, s in enumerate(summaries):
        chunk_content = _sanitize_text(chunks[i].content) if i < len(chunks) else s.summary_text
        tf, dl = compute_tf(chunk_content)
        rows.append({
            "content": chunk_content,
            "embedding": s.embedding,
            "summary": s.summary_text,
            "tfidf": tf,
            "doc_length": dl,
            "provenance": chunks[i].provenance if i < len(chunks) else {"source_url": source_url},
            "valid_from": s.temporal_context,
        })

    # Phase 4: Batch insert all SUMMARY nodes in single transaction
    summary_ids = bulk_insert_summaries(rows, project_id)
    log.info("Inserted %d SUMMARY nodes (batched)", len(summary_ids))

    # Phase 5: PART_OF edges
    for sid in summary_ids:
        _safe_insert_edge(sid, project_id, "PART_OF")

    # Phase 6: MMR link (optional, off by default)
    mmr_edges = 0
    if MMR_ON_INGEST:
        for sid, emb in zip(summary_ids, embeddings):
            mmr_edges += mmr_link_summary(sid, emb, project_id)

    # Phase 7: Project summary (first time only)
    _generate_project_summary(project_id, chunks)

    return {
        "summaries": len(summary_ids),
        "project_id": str(project_id),
        "temporal_nodes": sum(1 for s in summaries if s.temporal_context),
        "mmr_edges": mmr_edges,
    }
