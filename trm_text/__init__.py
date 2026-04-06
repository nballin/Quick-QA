"""
Text → grid representation and encoders for TRM-style QA integration.

- representation: map Unicode text to fixed H×W integer grids (byte-level).
- grid_encoder: small CNN + pool → dense vector (trainable surrogate).
- trm_arc_adapter: load ARC TRM checkpoint + byte remap + query puzzle ids + pooled hidden states.
- train_contrastive: batch contrastive loss for retrieval fine-tuning.
- hooks.load_trm_from_arc_checkpoint: convenience loader.
"""

from trm_text.representation import TextGridConfig, text_to_grid, texts_to_batch
from trm_text.grid_encoder import TextGridEncoder
from trm_text import hooks

__all__ = [
    "TextGridConfig",
    "text_to_grid",
    "texts_to_batch",
    "TextGridEncoder",
    "hooks",
]
