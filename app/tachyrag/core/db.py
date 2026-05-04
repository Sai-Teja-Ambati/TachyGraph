from __future__ import annotations

import uuid
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from tachyrag.config import DATABASE_URL, POOL_MAX, POOL_MIN


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    conn.autocommit = False


pool = ConnectionPool(
    DATABASE_URL,
    min_size=POOL_MIN,
    max_size=POOL_MAX,
    configure=_configure,
    kwargs={"row_factory": dict_row},
)


def insert_node(
    label: str,
    content: str,
    project_id: uuid.UUID,
    *,
    embedding: list[float] | None = None,
    summary: str | None = None,
    tfidf: dict | None = None,
    doc_length: int = 0,
    cluster_id: uuid.UUID | None = None,
    confidence: float | None = None,
    degree_cap: int = 10,
    provenance: dict | None = None,
    valid_from: str | None = None,
) -> uuid.UUID:
    with pool.connection() as conn, conn.cursor() as cur:
        if valid_from:
            cur.execute(
                """
                INSERT INTO nodes (label, content, embedding, summary, tfidf, doc_length,
                                   cluster_id, confidence, degree_cap, provenance, project_id, valid_from)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s,
                        %s, %s, %s, %s::jsonb, %s, %s::timestamptz)
                RETURNING id
                """,
                (
                    label, content, embedding, summary,
                    psycopg.types.json.Jsonb(tfidf) if tfidf else None,
                    doc_length,
                    cluster_id, confidence, degree_cap,
                    psycopg.types.json.Jsonb(provenance) if provenance else None,
                    project_id, valid_from,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO nodes (label, content, embedding, summary, tfidf, doc_length,
                                   cluster_id, confidence, degree_cap, provenance, project_id)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s,
                        %s, %s, %s, %s::jsonb, %s)
                RETURNING id
                """,
                (
                    label, content, embedding, summary,
                    psycopg.types.json.Jsonb(tfidf) if tfidf else None,
                    doc_length,
                    cluster_id, confidence, degree_cap,
                    psycopg.types.json.Jsonb(provenance) if provenance else None,
                    project_id,
                ),
            )
        node_id = cur.fetchone()["id"]

        # Update BM25 global stats
        if tfidf:
            _update_bm25_df(cur, tfidf, doc_length)

        conn.commit()
        return node_id


def insert_edge(
    source_id: uuid.UUID,
    target_id: uuid.UUID,
    label: str,
    weight: float = 1.0,
) -> uuid.UUID:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO edges (source_id, target_id, label, weight)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (source_id, target_id, label, weight),
        )
        edge_id = cur.fetchone()["id"]
        conn.commit()
        return edge_id


def get_node(node_id: uuid.UUID) -> dict[str, Any] | None:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM nodes WHERE id = %s", (node_id,))
        return cur.fetchone()


def _update_bm25_df(cur, tfidf: dict, doc_length: int) -> None:
    """Increment global DF counters and update corpus stats."""
    for term in tfidf:
        cur.execute(
            """
            INSERT INTO bm25_df (term, doc_count) VALUES (%s, 1)
            ON CONFLICT (term) DO UPDATE SET doc_count = bm25_df.doc_count + 1
            """,
            (term,),
        )
    cur.execute(
        """
        UPDATE bm25_stats SET
            total_docs = total_docs + 1,
            avg_doc_length = (avg_doc_length * total_docs + %s) / (total_docs + 1)
        WHERE id = 1
        """,
        (doc_length,),
    )


def bulk_insert_nodes(rows: list[dict]) -> list[uuid.UUID]:
    """Insert multiple nodes in a single transaction."""
    ids = []
    with pool.connection() as conn, conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """
                INSERT INTO nodes (label, content, embedding, summary, tfidf, doc_length,
                                   confidence, degree_cap, provenance, project_id)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s)
                RETURNING id
                """,
                (
                    r["label"], r["content"], r.get("embedding"),
                    r.get("summary"),
                    psycopg.types.json.Jsonb(r["tfidf"]) if r.get("tfidf") else None,
                    r.get("doc_length", 0),
                    r.get("confidence"), r.get("degree_cap", 10),
                    psycopg.types.json.Jsonb(r["provenance"]) if r.get("provenance") else None,
                    r["project_id"],
                ),
            )
            ids.append(cur.fetchone()["id"])
            if r.get("tfidf"):
                _update_bm25_df(cur, r["tfidf"], r.get("doc_length", 0))
        conn.commit()
    return ids





def bulk_insert_summaries(rows: list[dict], project_id: uuid.UUID) -> list[uuid.UUID]:
    """Batch insert SUMMARY nodes — single transaction, single BM25 stats update."""
    ids = []
    total_doc_length = 0
    with pool.connection() as conn, conn.cursor() as cur:
        for r in rows:
            if r.get("valid_from"):
                cur.execute(
                    """
                    INSERT INTO nodes (label, content, embedding, summary, tfidf, doc_length,
                                       degree_cap, provenance, project_id, valid_from)
                    VALUES ('SUMMARY', %s, %s, %s, %s::jsonb, %s, 10, %s::jsonb, %s, %s::timestamptz)
                    RETURNING id
                    """,
                    (
                        r["content"], r["embedding"], r["summary"],
                        psycopg.types.json.Jsonb(r["tfidf"]) if r.get("tfidf") else None,
                        r.get("doc_length", 0),
                        psycopg.types.json.Jsonb(r["provenance"]) if r.get("provenance") else None,
                        project_id, r["valid_from"],
                    ),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO nodes (label, content, embedding, summary, tfidf, doc_length,
                                       degree_cap, provenance, project_id)
                    VALUES ('SUMMARY', %s, %s, %s, %s::jsonb, %s, 10, %s::jsonb, %s)
                    RETURNING id
                    """,
                    (
                        r["content"], r["embedding"], r["summary"],
                        psycopg.types.json.Jsonb(r["tfidf"]) if r.get("tfidf") else None,
                        r.get("doc_length", 0),
                        psycopg.types.json.Jsonb(r["provenance"]) if r.get("provenance") else None,
                        project_id,
                    ),
                )
            ids.append(cur.fetchone()["id"])
            if r.get("tfidf"):
                for term in r["tfidf"]:
                    cur.execute(
                        "INSERT INTO bm25_df (term, doc_count) VALUES (%s, 1) ON CONFLICT (term) DO UPDATE SET doc_count = bm25_df.doc_count + 1",
                        (term,),
                    )
            total_doc_length += r.get("doc_length", 0)
        if rows:
            cur.execute(
                "UPDATE bm25_stats SET total_docs = total_docs + %s, avg_doc_length = CASE WHEN total_docs + %s > 0 THEN (avg_doc_length * total_docs + %s) / (total_docs + %s) ELSE 0 END WHERE id = 1",
                (len(rows), len(rows), total_doc_length, len(rows)),
            )
        conn.commit()
    return ids


def ensure_project(project_id: uuid.UUID, name: str) -> uuid.UUID:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM nodes WHERE id = %s AND label = 'PROJECT'", (project_id,))
        row = cur.fetchone()
        if row:
            return row["id"]
        cur.execute(
            """
            INSERT INTO nodes (id, label, content, degree_cap, project_id)
            VALUES (%s, 'PROJECT', %s, 0, %s)
            RETURNING id
            """,
            (project_id, name, project_id),
        )
        node_id = cur.fetchone()["id"]
        conn.commit()
        return node_id


def update_project_summary(project_id: uuid.UUID, summary: str) -> None:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE nodes SET summary = %s WHERE id = %s AND label = 'PROJECT'",
            (summary, project_id),
        )
        conn.commit()





def update_node_summary(node_id: uuid.UUID, summary: str) -> None:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE nodes SET summary = %s WHERE id = %s", (summary, node_id))
        conn.commit()


def get_all_projects() -> list[dict]:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT n.id, n.content AS name, n.summary,
                   (SELECT COUNT(*) FROM nodes c WHERE c.project_id = n.id AND c.label != 'PROJECT') AS node_count
            FROM nodes n
            WHERE n.label = 'PROJECT'
            ORDER BY n.created_at
            """,
        )
        return cur.fetchall()


def match_project_by_name(name: str) -> uuid.UUID | None:
    """Fuzzy match a project by name. Returns project_id or None."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM nodes WHERE label = 'PROJECT' AND LOWER(content) = LOWER(%s) LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
        if row:
            return row["id"]
        cur.execute(
            "SELECT id, content FROM nodes WHERE label = 'PROJECT' AND LOWER(content) LIKE LOWER(%s) LIMIT 1",
            (f"%{name}%",),
        )
        row = cur.fetchone()
        return row["id"] if row else None
