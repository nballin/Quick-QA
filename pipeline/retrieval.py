"""Hybrid FAISS + BM25 retrieval with RRF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pipeline.chunker import Chunk
from pipeline.encoder import BaseEncoder


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    source: str


class Retriever:
    def __init__(
        self,
        faiss_index: Any,
        bm25_index: Any | None,
        passage_store: Any,
        encoder: BaseEncoder,
        config: dict[str, Any],
    ) -> None:
        self._faiss = faiss_index
        self._bm25 = bm25_index
        self._store = passage_store
        self._encoder = encoder
        self._cfg = config

    def retrieve(
        self,
        query: str,
        top_k_faiss: int,
        top_k_bm25: int,
        use_hybrid: bool,
    ) -> list[RetrievalResult]:
        ret_cfg = self._cfg.get("retrieval", {})
        faiss_w = float(ret_cfg.get("faiss_weight", 0.7))
        bm25_w = float(ret_cfg.get("bm25_weight", 0.3))
        k_rrf = 60

        qvec = self._encoder.encode([query], batch_size=1, show_progress=False)[0]

        if self._faiss.ntotal == 0:
            return []

        fk = min(top_k_faiss, self._faiss.ntotal)
        scores_f, idx_f = self._faiss.search(qvec, fk)
        faiss_order: list[int] = [int(i) for i in idx_f if i >= 0]

        rrf_faiss: dict[int, float] = {}
        for rank, idx in enumerate(faiss_order):
            rrf_faiss[idx] = 1.0 / (k_rrf + rank + 1)

        if not use_hybrid or self._bm25 is None:
            out: list[RetrievalResult] = []
            for rank, idx in enumerate(faiss_order):
                sc = float(scores_f[rank]) if rank < len(scores_f) else rrf_faiss[idx]
                out.append(
                    RetrievalResult(
                        chunk=self._store.get(idx),
                        score=sc,
                        source="faiss",
                    )
                )
            return out

        bm25_pairs = self._bm25.search(query, top_k_bm25)
        bm25_order = [i for _, i in bm25_pairs]
        rrf_bm25: dict[int, float] = {}
        for rank, idx in enumerate(bm25_order):
            rrf_bm25[idx] = 1.0 / (k_rrf + rank + 1)

        all_idx = set(faiss_order) | set(bm25_order)
        combined: dict[int, float] = {}
        for idx in all_idx:
            combined[idx] = faiss_w * rrf_faiss.get(idx, 0.0) + bm25_w * rrf_bm25.get(idx, 0.0)

        ranked = sorted(combined.keys(), key=lambda i: -combined[i])
        return [
            RetrievalResult(
                chunk=self._store.get(idx),
                score=combined[idx],
                source="hybrid",
            )
            for idx in ranked
        ]
