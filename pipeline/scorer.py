"""Cosine re-ranking (dot product on L2-normalized embeddings)."""

from __future__ import annotations

from typing import Any

import numpy as np

from pipeline.encoder import BaseEncoder
from pipeline.retrieval import RetrievalResult


class Scorer:
    def __init__(self, encoder: BaseEncoder) -> None:
        self._encoder = encoder

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not candidates:
            return []
        texts = [r.chunk.text for r in candidates]
        qv = self._encoder.encode([query], batch_size=1, show_progress=False)
        pv = self._encoder.encode(texts, batch_size=min(32, len(texts)), show_progress=False)
        sims = (qv @ pv.T)[0]
        order = np.argsort(-sims)[:top_k]
        out: list[RetrievalResult] = []
        for i in order:
            r = candidates[int(i)]
            out.append(
                RetrievalResult(
                    chunk=r.chunk,
                    score=float(sims[int(i)]),
                    source="reranked",
                )
            )
        return out
