from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import faiss
import numpy as np

from tachyrag.config import (
    EMBEDDING_DIM,
    FAISS_GPU_ID,
    FAISS_GPU_MEMORY,
    FAISS_INDEX_DIR,
    FAISS_M,
    FAISS_NBITS,
    FAISS_NLIST,
    FAISS_NPROBE,
    FAISS_USE_GPU,
)

log = logging.getLogger(__name__)


class UnifiedVectorIndex:
    """Unified FAISS interface: CPU default, GPU auto-detect with fallback."""

    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        nlist: int = FAISS_NLIST,
        m: int = FAISS_M,
        nbits: int = FAISS_NBITS,
        gpu_id: int = FAISS_GPU_ID,
        use_gpu: bool = FAISS_USE_GPU,
        gpu_memory: int = FAISS_GPU_MEMORY,
    ):
        self.dim = dim
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.gpu_id = gpu_id
        self.use_gpu = use_gpu

        self.gpu_available = self._check_gpu_available() if use_gpu else False
        if use_gpu and not self.gpu_available:
            warnings.warn("GPU requested but not available. Falling back to CPU.")
            self.use_gpu = False

        self.res = None
        if self.use_gpu:
            self.res = faiss.StandardGpuResources()
            self.res.setTempMemory(gpu_memory * 1024 * 1024)

        self.index = self._create_index()
        self.is_trained = False
        self.id_map: dict[int, str] = {}
        self.reverse_map: dict[str, int] = {}
        self._next_id = 0

        mode = "GPU" if self.use_gpu else "CPU"
        log.info("UnifiedVectorIndex initialized (%s, dim=%d, nlist=%d)", mode, dim, nlist)

    @staticmethod
    def _check_gpu_available() -> bool:
        try:
            res = faiss.StandardGpuResources()
            test = faiss.IndexFlatL2(8)
            gpu_test = faiss.index_cpu_to_gpu(res, 0, test)
            del gpu_test, res
            return True
        except (AttributeError, RuntimeError, ImportError):
            return False

    def _create_index(self, flat: bool = False) -> faiss.Index:
        if flat:
            cpu_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        else:
            quantizer = faiss.IndexFlatIP(self.dim)
            cpu_index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.m, self.nbits)
        if self.use_gpu:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            return faiss.index_cpu_to_gpu(self.res, self.gpu_id, cpu_index, co)
        return cpu_index

    def train(self, vectors: np.ndarray) -> None:
        vectors = vectors.astype("float32")
        min_required = self.nlist * 30
        if len(vectors) < min_required:
            log.info("Too few vectors (%d < %d) for IVF-PQ, using flat index", len(vectors), min_required)
            self.index = self._create_index(flat=True)
            self.is_trained = True
            return
        self.index.train(vectors)
        self.is_trained = True
        log.info("FAISS index trained on %d vectors", len(vectors))

    def add_with_ids(self, vectors: np.ndarray, uuids: list[str]) -> None:
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
        vectors = vectors.astype("float32")
        faiss_ids = np.arange(self._next_id, self._next_id + len(uuids), dtype="int64")
        for fid, uid in zip(faiss_ids, uuids):
            self.id_map[int(fid)] = uid
            self.reverse_map[uid] = int(fid)
        self.index.add_with_ids(vectors, faiss_ids)
        self._next_id += len(uuids)

    def search(
        self,
        query: np.ndarray,
        k: int = 21,
        nprobe: int = FAISS_NPROBE,
    ) -> tuple[np.ndarray, list[str | None]]:
        query = query.astype("float32").reshape(1, -1)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe
        distances, faiss_ids = self.index.search(query, k)
        uuids = [self.id_map.get(int(fid)) for fid in faiss_ids[0]]
        return distances[0], uuids

    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 21,
        nprobe: int = FAISS_NPROBE,
    ) -> tuple[np.ndarray, list[list[str | None]]]:
        queries = queries.astype("float32")
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe
        distances, faiss_ids = self.index.search(queries, k)
        all_uuids = [
            [self.id_map.get(int(fid)) for fid in batch] for batch in faiss_ids
        ]
        return distances, all_uuids

    def remove_ids(self, uuids: list[str]) -> int:
        faiss_ids = []
        for uid in uuids:
            if uid in self.reverse_map:
                fid = self.reverse_map[uid]
                faiss_ids.append(fid)
                del self.id_map[fid]
                del self.reverse_map[uid]
        if faiss_ids:
            self.index.remove_ids(np.array(faiss_ids, dtype="int64"))
        return len(faiss_ids)

    def save(self, path: str | None = None) -> None:
        p = Path(path or os.path.join(FAISS_INDEX_DIR, "index.faiss"))
        p.parent.mkdir(parents=True, exist_ok=True)
        cpu_index = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
        faiss.write_index(cpu_index, str(p))
        np.savez(
            str(p.with_suffix(".mapping.npz")),
            id_map=self.id_map,
            reverse_map=self.reverse_map,
            next_id=self._next_id,
        )
        log.info("FAISS index saved to %s", p)

    def load(self, path: str | None = None) -> None:
        p = Path(path or os.path.join(FAISS_INDEX_DIR, "index.faiss"))
        if not p.exists():
            log.warning("No FAISS index at %s, starting fresh", p)
            return
        cpu_index = faiss.read_index(str(p))
        if self.use_gpu:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            self.index = faiss.index_cpu_to_gpu(self.res, self.gpu_id, cpu_index, co)
        else:
            self.index = cpu_index
        self.is_trained = True
        mapping_path = p.with_suffix(".mapping.npz")
        if mapping_path.exists():
            data = np.load(str(mapping_path), allow_pickle=True)
            self.id_map = data["id_map"].item()
            self.reverse_map = data["reverse_map"].item()
            self._next_id = int(data["next_id"])
        log.info("FAISS index loaded from %s (%d vectors)", p, self.ntotal)

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def __repr__(self) -> str:
        mode = "GPU" if self.use_gpu else "CPU"
        return f"UnifiedVectorIndex({mode}, dim={self.dim}, ntotal={self.ntotal})"


def create_vector_index(use_gpu: bool = FAISS_USE_GPU, **kwargs) -> UnifiedVectorIndex:
    """Factory: create index with auto GPU detection and CPU fallback."""
    return UnifiedVectorIndex(use_gpu=use_gpu, **kwargs)
