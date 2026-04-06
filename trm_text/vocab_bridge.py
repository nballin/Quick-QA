"""
Map byte-level grid token ids (0=pad, 1..256 = bytes) to ARC TRM token ids.

The ARC model expects ids in [0, vocab_size). This module provides:
- deterministic remap (runs without training; not semantically meaningful)
- learnable remap (nn.Module) for fine-tuning
"""

from __future__ import annotations

import torch
from torch import nn


def remap_bytes_deterministic(
    byte_ids: torch.Tensor,
    vocab_size: int,
    pad_id: int = 0,
) -> torch.Tensor:
    """
    Map byte ids to ARC vocab range without extra parameters.

    Preserves pad_id. Other ids spread across non-pad tokens (avoids 0 where possible).
    """
    x = byte_ids.long()
    mask = x != pad_id
    denom = max(1, vocab_size - 1)
    mapped = (x % denom) + 1
    return torch.where(mask, mapped, torch.full_like(x, pad_id)).clamp(0, vocab_size - 1)


class LearnedByteToVocab(nn.Module):
    """
    Learned logits per byte id over TRM vocab; forward uses argmax (train logits with auxiliary loss).

    Initialize near deterministic remap so a cold start still runs.
    """

    def __init__(self, vocab_size: int, num_byte_ids: int = 257) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_byte_ids, vocab_size))
        with torch.no_grad():
            for b in range(num_byte_ids):
                self.logits[b, (b % max(1, vocab_size - 1)) + 1] = 2.0

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        idx = byte_ids.clamp(0, self.logits.size(0) - 1)
        return torch.argmax(self.logits[idx], dim=-1)
