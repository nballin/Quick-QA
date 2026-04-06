"""
Hooks for wiring ARC TinyRecursiveModels checkpoints into QA text grids.
"""

from __future__ import annotations

from typing import Any, Optional

from trm_text.trm_arc_adapter import TrmArcTextEncoder, build_encoder_from_paths


def load_trm_from_arc_checkpoint(
    checkpoint_path: str,
    tiny_recursive_models_root: str,
    dataset_json_path: Optional[str] = None,
    learned_bridge: bool = False,
) -> TrmArcTextEncoder:
    """
    Load TRM weights and return a `TrmArcTextEncoder` for text pairs.

    `dataset_json_path` must be the `dataset.json` from the **same** ARC dataset
    build as the checkpoint (matching seq_len, vocab_size, num_puzzle_identifiers).
    If omitted, uses `trm_text/fixtures/example_arc_dataset.json` (typical ARC-AGI layout;
    only use if your checkpoint matches those numbers).
    """
    import os

    root = os.path.dirname(os.path.abspath(__file__))
    default_json = os.path.join(root, "fixtures", "example_arc_dataset.json")
    meta_path = dataset_json_path or default_json
    return build_encoder_from_paths(
        checkpoint_path=checkpoint_path,
        dataset_json_path=meta_path,
        tiny_recursive_models_root=tiny_recursive_models_root,
        learned_bridge=learned_bridge,
    )


def describe_integration_steps() -> str:
    return (
        "1) Build ARC data once and keep its dataset.json next to your workflow.\n"
        "2) Fetch checkpoint: python -m experiments.fetch_hf_checkpoints (in TinyRecursiveModels).\n"
        "3) Pass checkpoint path + dataset.json + TinyRecursiveModels root to load_trm_from_arc_checkpoint.\n"
        "4) Fine-tune LearnedByteToVocab + optional TRM layers on QA/retrieval data."
    )
