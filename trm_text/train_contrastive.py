"""
Contrastive / retrieval losses for pooled embeddings (TRM or surrogate).

Example: in-batch negatives with dot-product softmax (SimCLR-style).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def batch_contrastive_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    emb_a, emb_b: (B, D), paired rows must match.
    Symmetric InfoNCE using batch as negatives.
    """
    a = F.normalize(emb_a, dim=-1)
    b = F.normalize(emb_b, dim=-1)
    logits = (a @ b.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )
