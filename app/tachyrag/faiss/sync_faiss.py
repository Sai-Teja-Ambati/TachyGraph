from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tachyrag.config import FAISS_SYNC_BATCH, FAISS_SYNC_INTERVAL
from tachyrag.core.db import pool
from tachyrag.faiss.vector_index import UnifiedVectorIndex

log = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1)


class FaissSyncService:
    def __init__(self, faiss_index: UnifiedVectorIndex, batch_size: int = FAISS_SYNC_BATCH):
        self.faiss = faiss_index
        self.batch_size = batch_size
        self.pending_adds: list[tuple[str, np.ndarray]] = []
        self.pending_removes: set[str] = set()
        self._task: asyncio.Task | None = None

    def start_background(self) -> None:
        self._task = asyncio.create_task(self._loop())
        log.info("FAISS sync background task started (interval=%ds)", FAISS_SYNC_INTERVAL)

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(FAISS_SYNC_INTERVAL)
            try:
                await asyncio.get_event_loop().run_in_executor(_executor, self._flush_sync)
            except Exception as e:
                log.error("FAISS sync flush error: %s", e)

    def _flush_sync(self) -> None:
        """Synchronous flush for running in executor."""
        removed = 0
        added = 0

        if self.pending_removes:
            removed = self.faiss.remove_ids(list(self.pending_removes))
            self.pending_removes.clear()

        if self.pending_adds:
            ids, vectors = zip(*self.pending_adds)
            vectors_np = np.stack(vectors).astype("float32")
            self.faiss.add_with_ids(vectors_np, list(ids))
            added = len(ids)
            self.pending_adds.clear()

        if added or removed:
            log.info("FAISS sync: +%d added, -%d removed (total=%d)", added, removed, self.faiss.ntotal)

    def queue_add(self, node_id: str, embedding: np.ndarray) -> None:
        self.pending_adds.append((node_id, embedding))
        if len(self.pending_adds) >= self.batch_size:
            asyncio.create_task(self.flush())

    def queue_remove(self, node_id: str) -> None:
        self.pending_removes.add(node_id)

    async def flush(self) -> None:
        await asyncio.get_event_loop().run_in_executor(_executor, self._flush_sync)

    def full_sync(self, project_id: uuid.UUID | None = None) -> int:
        """Full sync from pgvector to FAISS. Returns vectors synced."""
        query = """
            SELECT id, embedding FROM nodes
            WHERE embedding IS NOT NULL
              AND (valid_until IS NULL OR valid_until > NOW())
        """
        params: tuple = ()
        if project_id:
            query += " AND project_id = %s"
            params = (project_id,)

        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        if not rows:
            return 0

        ids = [str(r["id"]) for r in rows]
        vectors = np.array([r["embedding"] for r in rows], dtype="float32")

        if not self.faiss.is_trained or self.faiss.ntotal == 0:
            self.faiss.train(vectors)

        self.faiss.add_with_ids(vectors, ids)
        log.info("FAISS full sync: %d vectors loaded", len(ids))
        return len(ids)
