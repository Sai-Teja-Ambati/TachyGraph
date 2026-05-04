from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from tachyrag.core.db import pool
from tachyrag.search.search import search


@dataclass
class Fact:
    node_id: uuid.UUID
    summary: str
    provenance: dict | None
    relationship: str | None = None


@dataclass
class FactChain:
    facts: list[Fact] = field(default_factory=list)
    complete: bool = True

    @property
    def empty(self) -> bool:
        return len(self.facts) == 0


def _traverse_edges(node_id: uuid.UUID, depth: int = 2) -> list[dict]:
    """Walk outgoing edges up to `depth` hops."""
    visited = set()
    frontier = [node_id]
    edges_found = []

    for _ in range(depth):
        if not frontier:
            break
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.label AS rel, e.target_id,
                       t.content, t.summary, t.provenance
                FROM edges e
                JOIN nodes t ON t.id = e.target_id
                WHERE e.source_id = ANY(%s)
                  AND e.target_id != ALL(%s)
                  AND (t.valid_until IS NULL OR t.valid_until > NOW())
                """,
                (frontier, list(visited)),
            )
            rows = cur.fetchall()

        next_frontier = []
        for r in rows:
            visited.add(r["target_id"])
            edges_found.append(r)
            next_frontier.append(r["target_id"])
        frontier = next_frontier

    return edges_found


def build_response(
    query_text: str,
    project_id: uuid.UUID,
    k: int = 5,
) -> FactChain:
    results = search(query_text, project_id, k=k)

    if not results:
        return FactChain(facts=[], complete=False)

    chain = FactChain()
    for r in results:
        chain.facts.append(Fact(
            node_id=r["id"],
            summary=r.get("summary") or r["content"],
            provenance=r.get("provenance"),
        ))

        # Traverse edges for supporting facts
        edges = _traverse_edges(r["id"])
        for e in edges:
            chain.facts.append(Fact(
                node_id=e["target_id"],
                summary=e.get("summary") or e["content"],
                provenance=e.get("provenance"),
                relationship=e["rel"],
            ))

    if not chain.facts:
        chain.complete = False

    return chain


def format_response(chain: FactChain) -> str:
    if chain.empty:
        return "Data missing"

    lines = []
    for f in chain.facts:
        prefix = f"[{f.relationship}] " if f.relationship else ""
        source = f.provenance.get("source_url", "unknown") if f.provenance else "unknown"
        lines.append(f"{prefix}{f.summary} (source: {source})")
    return "\n".join(lines)
