"""
Load ARC TinyRecursiveModels TRM weights and run text (byte-grid) inputs.

Requires:
  - `TinyRecursiveModels` checkout (set TRM_ROOT or pass `tiny_recursive_models_root`)
  - A `dataset.json` from an ARC build (same seq_len / vocab / num_puzzle_identifiers as the checkpoint)
  - Checkpoint `.pt` file (e.g. from arcprize/trm_arc_prize_verification)

This wires vocabulary remapping + query-hashed puzzle ids + pooled hidden states for downstream QA.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

import torch
import yaml

from trm_text.representation import TextGridConfig, texts_to_batch
from trm_text.vocab_bridge import LearnedByteToVocab, remap_bytes_deterministic
from trm_text.query_conditioning import batch_query_puzzle_ids


@dataclass
class ArcMetadata:
    seq_len: int
    vocab_size: int
    pad_id: int
    num_puzzle_identifiers: int
    blank_identifier_id: int = 0


def load_arc_metadata(dataset_json_path: str) -> ArcMetadata:
    with open(dataset_json_path, encoding="utf-8") as f:
        d = json.load(f)
    return ArcMetadata(
        seq_len=int(d["seq_len"]),
        vocab_size=int(d["vocab_size"]),
        pad_id=int(d["pad_id"]),
        num_puzzle_identifiers=int(d["num_puzzle_identifiers"]),
        blank_identifier_id=int(d.get("blank_identifier_id", 0)),
    )


def _ensure_trm_import_path(tiny_recursive_models_root: str) -> None:
    root = os.path.abspath(tiny_recursive_models_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_arch_config(trm_root: str) -> dict:
    arch_path = os.path.join(trm_root, "config", "arch", "trm.yaml")
    with open(arch_path, encoding="utf-8") as f:
        arch = yaml.safe_load(f)
    arch.pop("name", None)
    arch.pop("loss", None)
    # Resolve Hydra-style self-reference from the same file
    hs = arch.get("hidden_size")
    if isinstance(arch.get("puzzle_emb_ndim"), str) and "${" in str(arch.get("puzzle_emb_ndim")):
        arch["puzzle_emb_ndim"] = hs
    return arch


def _merge_model_cfg(
    arch: dict,
    meta: ArcMetadata,
    batch_size: int,
) -> dict:
    inner = {k: v for k, v in arch.items() if k not in ("name", "loss")}
    inner["batch_size"] = batch_size
    inner["vocab_size"] = meta.vocab_size
    inner["seq_len"] = meta.seq_len
    inner["num_puzzle_identifiers"] = meta.num_puzzle_identifiers
    inner["causal"] = False
    return inner


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    """Map checkpoint keys onto TinyRecursiveReasoningModel_ACTV1 (expects inner.*)."""
    out = {}
    for k, v in state_dict.items():
        key = k
        while key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        while key.startswith("model."):
            key = key[len("model.") :]
        if not key.startswith("inner."):
            key = "inner." + key
        out[key] = v
    return out


def load_trm_actv1(
    checkpoint_path: str,
    dataset_json_path: str,
    tiny_recursive_models_root: str,
    batch_size: int = 4,
    device: Optional[str] = None,
) -> Any:
    """
    Return `TinyRecursiveReasoningModel_ACTV1` with weights loaded (no torch.compile).
    """
    _ensure_trm_import_path(tiny_recursive_models_root)
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    meta = load_arc_metadata(dataset_json_path)
    arch_yaml = _load_arch_config(tiny_recursive_models_root)
    inner_cfg = _merge_model_cfg(arch_yaml, meta, batch_size=batch_size)
    os.environ.setdefault("DISABLE_COMPILE", "1")

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRecursiveReasoningModel_ACTV1(inner_cfg)
    try:
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        sd = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if not isinstance(sd, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(sd)}")
    sd = _normalize_state_dict_keys(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[trm_arc_adapter] load_state_dict missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"[trm_arc_adapter] load_state_dict unexpected keys (first 10): {unexpected[:10]}")
    model.to(dev)
    model.eval()
    return model


def _flatten_grid_to_seq(
    grids: torch.Tensor,
    seq_len: int,
    pad_id: int,
) -> torch.Tensor:
    """grids: (B, H, W) long -> (B, seq_len) long."""
    b = grids.shape[0]
    flat = grids.reshape(b, -1).long()
    if flat.shape[1] < seq_len:
        pad = torch.full((b, seq_len - flat.shape[1]), pad_id, dtype=flat.dtype, device=flat.device)
        flat = torch.cat([flat, pad], dim=1)
    else:
        flat = flat[:, :seq_len]
    return flat


class TrmArcTextEncoder(torch.nn.Module):
    """
    End-to-end: UTF-8 text -> byte grid -> ARC vocab ids -> TRM ACT forward -> pooled embedding.

    Use `learned_bridge=True` to train the byte→vocab mapping; otherwise deterministic remap.
    """

    def __init__(
        self,
        trm_model: Any,
        arc_meta: ArcMetadata,
        text_grid_config: Optional[TextGridConfig] = None,
        learned_bridge: bool = False,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        self.trm = trm_model
        self.meta = arc_meta
        self.text_cfg = text_grid_config or TextGridConfig()
        vs = arc_meta.vocab_size
        self.bridge_module: Optional[LearnedByteToVocab] = (
            LearnedByteToVocab(vs) if learned_bridge else None
        )
        self.pool = pool

    def build_batch_from_texts(
        self,
        texts: list[str],
        queries: list[str],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if len(queries) != len(texts):
            raise ValueError("queries and texts must have same length (parallel query/passage pairs).")
        np_grids = texts_to_batch(texts, self.text_cfg)
        grids = torch.from_numpy(np_grids).to(device)
        flat_byte = _flatten_grid_to_seq(grids, self.meta.seq_len, self.meta.pad_id)
        if self.bridge_module is not None:
            inputs = self.bridge_module(flat_byte)
        else:
            inputs = remap_bytes_deterministic(
                flat_byte, self.meta.vocab_size, pad_id=self.meta.pad_id
            )
        pids = batch_query_puzzle_ids(queries, self.meta.num_puzzle_identifiers)
        puzzle_identifiers = torch.tensor(pids, dtype=torch.int32, device=device)
        return {
            "inputs": inputs.to(torch.int32),
            "puzzle_identifiers": puzzle_identifiers,
        }

    @torch.inference_mode()
    def encode_text_pairs(
        self,
        passages: list[str],
        queries: list[str],
    ) -> torch.Tensor:
        """
        One pooled vector per (query, passage) pair, shape (B, hidden_size).
        """
        device = next(self.trm.parameters()).device
        batch = self.build_batch_from_texts(passages, queries, device)
        return self._forward_encode(batch)

    def _forward_encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.trm.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        hidden: list[torch.Tensor] = []

        def hook(_m: Any, inp: tuple, _out: Any) -> None:
            hidden.append(inp[0].detach())

        h = self.trm.inner.lm_head.register_forward_hook(hook)
        try:
            carry = self.trm.initial_carry(batch)
            max_steps = int(os.environ.get("TRM_ENCODE_MAX_STEPS", str(self.trm.config.halt_max_steps)))
            max_steps = min(max_steps, int(self.trm.config.halt_max_steps))
            for _ in range(max_steps):
                carry, _outputs = self.trm(carry, batch)
                if bool(carry.halted.all()):
                    break
        finally:
            h.remove()
        if not hidden:
            raise RuntimeError("TRM forward did not capture hidden states.")
        z = hidden[-1]
        inner = self.trm.inner
        pe = int(getattr(inner, "puzzle_emb_len", self.trm.config.puzzle_emb_len))
        tok = z[:, pe:, :]
        if self.pool == "mean":
            return tok.mean(dim=1)
        if self.pool == "first":
            return tok[:, 0, :]
        raise ValueError(f"Unknown pool={self.pool}")


def build_encoder_from_paths(
    checkpoint_path: str,
    dataset_json_path: str,
    tiny_recursive_models_root: str,
    learned_bridge: bool = False,
) -> TrmArcTextEncoder:
    model = load_trm_actv1(
        checkpoint_path,
        dataset_json_path,
        tiny_recursive_models_root,
        batch_size=4,
    )
    meta = load_arc_metadata(dataset_json_path)
    return TrmArcTextEncoder(model, meta, learned_bridge=learned_bridge)
