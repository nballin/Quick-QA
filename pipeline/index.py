"""FAISS + BM25 + JSONL passage store."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)


def _chunk_to_dict(c: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "source": c.source,
        "text": c.text,
        "char_start": c.char_start,
        "char_end": c.char_end,
        "metadata": c.metadata,
    }


def _dict_to_chunk(d: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=d["chunk_id"],
        doc_id=d["doc_id"],
        source=d["source"],
        text=d["text"],
        char_start=int(d["char_start"]),
        char_end=int(d["char_end"]),
        metadata=d.get("metadata") or {},
    )


class PassageStore:
    """JSONL-backed store: line index == FAISS row index."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)

    def get(self, idx: int) -> Chunk:
        return self._chunks[idx]

    def __len__(self) -> int:
        return len(self._chunks)

    def all(self) -> list[Chunk]:
        return list(self._chunks)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(json.dumps(_chunk_to_dict(c), ensure_ascii=False) + "\n")
        logger.info("Saved %d passages to %s", len(self._chunks), p)

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Passage store not found: {p}. Run 'python main.py ingest' first.")
        self._chunks = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._chunks.append(_dict_to_chunk(json.loads(line)))
        logger.info("Loaded %d passages from %s", len(self._chunks), p)


class FaissIndex:
    def __init__(self, dim: int, use_gpu: bool = False) -> None:
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._use_gpu = use_gpu

    @classmethod
    def from_file(cls, path: str | Path) -> FaissIndex:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"FAISS index not found: {p}. Run 'python main.py ingest' first.")
        raw = faiss.read_index(str(p))
        inst = cls(dim=raw.d)
        inst._index = raw
        return inst

    def build(self, embeddings: np.ndarray) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self._index.reset()
        self._index.add(embeddings)

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        q = query_vec.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        k = min(top_k, max(1, self._index.ntotal))
        scores, indices = self._index.search(q, k)
        return scores[0], indices[0]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p))
        logger.info("Saved FAISS index to %s", p)

class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        tokenized = [c.text.lower().split() for c in chunks]
        self._chunks = chunks
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int) -> list[tuple[float, int]]:
        q = query.lower().split()
        scores = self._bm25.get_scores(q)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
        return [(float(s), int(i)) for i, s in ranked]


def build_index(
    chunks: list[Chunk],
    encoder: Any,
    config: dict[str, Any],
) -> tuple[FaissIndex, BM25Index | None, PassageStore]:
    """Encode chunks, build FAISS (and optionally BM25), save to disk."""
    paths = config["paths"]
    enc_cfg = config["encoder"]
    index_dir = Path(paths["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)

    texts = [c.text for c in chunks]
    bs = int(enc_cfg.get("batch_size", 32))
    logger.info("Encoding %d chunks for index...", len(texts))
    if not texts:
        raise ValueError("No chunk texts to encode.")
    all_emb: list[np.ndarray] = []
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        all_emb.append(encoder.encode(batch, batch_size=len(batch), show_progress=False))
        if (i // bs) % 10 == 0 and i + bs < len(texts):
            logger.info("  encoded %d / %d", min(i + bs, len(texts)), len(texts))
    embeddings = np.vstack(all_emb) if len(all_emb) > 1 else all_emb[0]

    dim = embeddings.shape[1]
    faiss_idx = FaissIndex(dim)
    faiss_idx.build(embeddings)

    store = PassageStore()
    store.add(chunks)
    faiss_path = Path(paths["faiss_index"])
    store_path = Path(paths["passage_store"])
    faiss_idx.save(faiss_path)
    store.save(store_path)

    bm25: BM25Index | None = None
    if config.get("retrieval", {}).get("use_bm25_hybrid", True):
        bm25 = BM25Index(chunks)
        logger.info("Built BM25 index over %d chunks", len(chunks))

    return faiss_idx, bm25, store


def load_index(config: dict[str, Any]) -> tuple[FaissIndex, BM25Index | None, PassageStore]:
    """Load FAISS, passage store, rebuild BM25 from stored chunks."""
    paths = config["paths"]
    store = PassageStore()
    store.load(paths["passage_store"])
    faiss_idx = FaissIndex.from_file(paths["faiss_index"])

    bm25: BM25Index | None = None
    if config.get("retrieval", {}).get("use_bm25_hybrid", True):
        bm25 = BM25Index(store.all())

    return faiss_idx, bm25, store
