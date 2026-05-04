from __future__ import annotations

import json
import logging
import uuid

from tachyrag.core.db import pool

log = logging.getLogger(__name__)


def export_json(project_id: uuid.UUID | None = None) -> dict:
    """Export nodes + edges as JSON (without embeddings)."""
    with pool.connection() as conn, conn.cursor() as cur:
        where = "WHERE project_id = %s" if project_id else ""
        params = (project_id,) if project_id else ()

        cur.execute(
            f"SELECT id, label, content, summary, tfidf, doc_length, confidence, valid_from, valid_until, provenance, project_id, created_at FROM nodes {where} ORDER BY created_at",
            params,
        )
        nodes = [
            {k: str(v) if isinstance(v, uuid.UUID) else (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()}
            for r in cur.fetchall()
        ]

        node_ids = [r["id"] for r in nodes]
        if node_ids:
            # Need UUIDs for the query
            raw_ids = [uuid.UUID(nid) if isinstance(nid, str) else nid for nid in node_ids]
            cur.execute(
                "SELECT id, source_id, target_id, label, weight, created_at FROM edges WHERE source_id = ANY(%s) OR target_id = ANY(%s)",
                (raw_ids, raw_ids),
            )
            edges = [
                {k: str(v) if isinstance(v, uuid.UUID) else (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()}
                for r in cur.fetchall()
            ]
        else:
            edges = []

    return {"nodes": nodes, "edges": edges, "count": {"nodes": len(nodes), "edges": len(edges)}}


def export_markdown(project_id: uuid.UUID | None = None) -> str:
    """Export all SUMMARY nodes as a markdown document grouped by source."""
    with pool.connection() as conn, conn.cursor() as cur:
        where = "WHERE label = 'SUMMARY' AND project_id = %s" if project_id else "WHERE label = 'SUMMARY'"
        params = (project_id,) if project_id else ()
        cur.execute(
            f"SELECT summary, content, provenance, created_at FROM nodes {where} ORDER BY created_at",
            params,
        )
        rows = cur.fetchall()

    if not rows:
        return "# Empty Knowledge Graph\n\nNo summaries found.\n"

    sections: dict[str, list] = {}
    for r in rows:
        source = r.get("provenance", {}).get("source_url", "unknown") if r.get("provenance") else "unknown"
        sections.setdefault(source, []).append(r)

    lines = ["# TachyGraph Knowledge Export\n"]
    for source, items in sections.items():
        lines.append(f"\n## Source: {source}\n")
        for item in items:
            lines.append(f"### {item.get('summary', 'Untitled')}\n")
            lines.append(item.get("content", "")[:2000])
            lines.append("")

    return "\n".join(lines)


def import_json(data: dict, project_id: uuid.UUID | None = None) -> dict:
    """Import a previously exported JSON backup. Re-ingests text (embeddings are model-specific)."""
    from tachyrag.ingest.ingestor import ingest_document
    from tachyrag.core.db import ensure_project

    nodes = data.get("nodes", [])
    summaries_only = [n for n in nodes if n.get("label") == "SUMMARY"]

    if not summaries_only:
        return {"imported": 0, "status": "no SUMMARY nodes found"}

    pid = project_id or uuid.uuid4()
    ensure_project(pid, "imported")

    imported = 0
    for n in summaries_only:
        try:
            ingest_document(
                text=n.get("content", ""),
                source_url=n.get("provenance", {}).get("source_url", "import://backup") if isinstance(n.get("provenance"), dict) else "import://backup",
                project_id=pid,
                project_name="imported",
            )
            imported += 1
        except Exception as e:
            log.warning("Import failed for node: %s", e)

    return {"imported": imported, "total": len(summaries_only), "project_id": str(pid)}
