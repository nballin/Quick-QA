"""
Map text to fixed-size 2D grids of small integer IDs (byte-oriented).

This matches the project brief: avoid BPE by encoding characters/bytes spatially.
ARC-TRM uses a different vocabulary; this grid is the *QA-side* input format you
fine-tune or distill into TRM-compatible tensors later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TextGridConfig:
    """Layout for packing UTF-8 text into a 2D grid."""

    height: int = 32
    width: int = 64
    pad_id: int = 0
    # Raw bytes map to ids 1..256; pad_id reserved for padding cells.

    @property
    def seq_len(self) -> int:
        return self.height * self.width


def _utf8_bytes(text: str) -> bytes:
    return text.encode("utf-8", errors="replace")


def byte_to_id(b: int) -> int:
    """Map a raw byte 0..255 to token id 1..256 (0 reserved for pad)."""
    return int(b) + 1


def text_to_grid(
    text: str,
    config: TextGridConfig | None = None,
) -> np.ndarray:
    """
    Pack UTF-8 bytes of `text` row-major into an (H, W) array of int64 ids.

    Long text is truncated; short text is padded with pad_id.
    """
    cfg = config or TextGridConfig()
    data = np.full((cfg.height, cfg.width), cfg.pad_id, dtype=np.int64)
    blob = _utf8_bytes(text)
    n = min(len(blob), cfg.seq_len)
    flat = data.ravel()
    for i in range(n):
        flat[i] = byte_to_id(blob[i])
    return data


def texts_to_batch(
    texts: list[str],
    config: TextGridConfig | None = None,
) -> np.ndarray:
    """Stack many texts to shape (B, H, W)."""
    cfg = config or TextGridConfig()
    return np.stack([text_to_grid(t, cfg) for t in texts], axis=0)


def concat_query_passage_grids(
    query: str,
    passage: str,
    config: TextGridConfig | None = None,
    separator: str = "\n---\n",
) -> np.ndarray:
    """
    Single grid containing query then separator then passage (same truncation rules).

    For query-conditioned passage encoding, prefer separate grids + two forward passes
    or a two-channel setup in a later encoder.
    """
    return text_to_grid(query + separator + passage, config)


if __name__ == "__main__":
    cfg = TextGridConfig(height=4, width=8)
    g = text_to_grid("Hello 世界", cfg)
    print(g)
    print("batch", texts_to_batch(["a", "bb"], cfg).shape)
