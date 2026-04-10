"""
Text encoders for QuickQA (retrieval + re-ranking).

What this module does
---------------------
Turns a list of strings into a single NumPy matrix of embeddings shaped (N, D).
The rest of the pipeline treats that as a black box: ingest/indexing, hybrid
retrieval (FAISS), and the scorer all call `encode()` without knowing which model
runs underneath.

Contract (BaseEncoder)
----------------------
- `encode(texts) -> ndarray` float32, shape (N, D). MiniLM outputs are L2-normalized;
  downstream code assumes vectors are comparable with inner product / cosine.
- `embedding_dim` is D (needed when building or validating the FAISS index).

Backends (`config.yaml` → `encoder.backend`)
--------------------------------------------
- `minilm` — SentenceTransformers model (default). Fully implemented; powers the
  current end-to-end CLI (ingest, query, eval).
- `trm` — ARC Tiny Recursive Model text path. Implementation lives in `trm_text/`
  (`TrmArcTextEncoder`, `hooks.load_trm_from_arc_checkpoint`). `TRMEncoder` here is
  a stub until it wraps that code and maps the generic `encode(texts)` API to
  TRM’s query–passage pairing and checkpoint paths.
- `cnn` — Lightweight grid CNN surrogate in `trm_text/grid_encoder.py`; same stub
  story as TRM.

Adding a backend: subclass `BaseEncoder`, implement `encode` + `embedding_dim`, and
register it in `build_encoder()`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class BaseEncoder(ABC):
    @abstractmethod
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Return (N, D) float32 L2-normalized embeddings."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        ...


class MiniLMEncoder(BaseEncoder):
    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        # Avoid SentenceTransformer internal tqdm (can segfault on some Python/torch builds).
        emb = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())


class TRMEncoder(BaseEncoder):
    """
    TRM integration — wire `trm_text.trm_arc_adapter.TrmArcTextEncoder` or
    `trm_text.hooks.load_trm_from_arc_checkpoint` and implement encode().
    """

    def __init__(self, **kwargs: object) -> None:
        pass

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        raise NotImplementedError(
            "TRM encoder integration — see trm_text/trm_arc_adapter.py and hooks.py"
        )

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError("TRM encoder integration")


class CNNEncoder(BaseEncoder):
    """CNN surrogate — see trm_text/grid_encoder.py (TextGridEncoder)."""

    def __init__(self, **kwargs: object) -> None:
        pass

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        raise NotImplementedError("CNN surrogate encoder — see trm_text/grid_encoder.py")

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError("CNN surrogate encoder")


def build_encoder(backend: str, **kwargs: object) -> BaseEncoder:
    """Instantiate the encoder named in config (`model_name`, `device`, etc. passed via kwargs)."""
    if backend == "minilm":
        return MiniLMEncoder(
            str(kwargs.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
            str(kwargs.get("device", "cpu")),
        )
    if backend == "trm":
        return TRMEncoder(**kwargs)
    if backend == "cnn":
        return CNNEncoder(**kwargs)
    raise ValueError(f"Unknown encoder backend: {backend}")
