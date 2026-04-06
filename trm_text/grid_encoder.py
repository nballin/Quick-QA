"""
Trainable encoder: integer grid -> dense vector.

Replace this module's core with TRM recursive blocks once weights and vocab are aligned.
"""

from __future__ import annotations

import torch
from torch import nn

from trm_text.representation import TextGridConfig


class TextGridEncoder(nn.Module):
    """
    Embedding lookup for grid ids + shallow CNN + GAP + linear.

    `vocab_size` must cover byte_to_id range: ids in [0, 256] with 0 = pad.
    """

    def __init__(
        self,
        config: TextGridConfig | None = None,
        vocab_size: int = 257,
        embed_dim: int = 64,
        out_dim: int = 384,
        hidden_channels: tuple[int, ...] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.config = config or TextGridConfig()
        self.out_dim = out_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        layers: list[nn.Module] = []
        c_in = embed_dim
        for c_out in hidden_channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))
            layers.append(nn.GELU())
            c_in = c_out
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(c_in, out_dim)
        self._ln = nn.LayerNorm(out_dim)

    def forward(self, grid_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_ids: (B, H, W) int64/long
        Returns:
            (B, out_dim) L2-normalized vectors (good for cosine retrieval).
        """
        x = self.embed(grid_ids.long())  # B,H,W,E
        x = x.permute(0, 3, 1, 2).contiguous()  # B,E,H,W
        x = self.cnn(x)
        x = x.mean(dim=(2, 3))  # global average pool
        x = self.proj(x)
        x = self._ln(x)
        return torch.nn.functional.normalize(x, dim=-1)
