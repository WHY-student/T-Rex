"""Checkpoint restoration helpers shared by T-Rex deployment entry points.

Post-training can change the action/state dimensions and wrap the action
expert's Linear layers with LoRA.  Those changes must be reconstructed before
``load_state_dict``; otherwise ``strict=False`` silently leaves a large part of
the action expert randomly initialized.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_LORA_TARGET_SUFFIXES = (
    "q_proj_action",
    "k_proj_action",
    "v_proj_action",
    "o_proj_action",
    "mlp_action.gate_proj",
    "mlp_action.up_proj",
    "mlp_action.down_proj",
)


class ActionLoRALinear(nn.Module):
    """Inference reconstruction of the LoRA wrapper used during post-training."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, value):
        base_output = self.base(value)
        lora_output = F.linear(
            F.linear(self.dropout(value), self.lora_A), self.lora_B
        )
        return base_output + lora_output * self.scaling


def inject_action_expert_lora(
    model: nn.Module, *, rank: int, alpha: float, dropout: float = 0.0
) -> int:
    """Wrap every action-expert Linear using the training-time module layout."""
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(ACTION_LORA_TARGET_SUFFIXES):
            parent_name, child_name = name.rsplit(".", 1)
            replacements.append((parent_name, child_name, module))

    module_lookup = dict(model.named_modules())
    for parent_name, child_name, module in replacements:
        setattr(
            module_lookup[parent_name],
            child_name,
            ActionLoRALinear(module, rank, alpha, dropout),
        )
    return len(replacements)


def load_checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a plain model state dict while using the safe torch loader when possible."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint is not a state dict: {path}")
    return dict(state)


def infer_action_lora_rank(state_dict: Mapping[str, torch.Tensor]) -> int | None:
    """Return the single LoRA rank encoded by checkpoint tensors, if present."""
    ranks = {
        int(value.shape[0])
        for key, value in state_dict.items()
        if key.endswith(".lora_A") and value.ndim == 2
    }
    if not ranks:
        return None
    if len(ranks) != 1:
        raise ValueError(f"Checkpoint contains inconsistent action LoRA ranks: {sorted(ranks)}")
    return ranks.pop()


def restore_action_lora(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    training_args: Mapping[str, Any],
    rank: int | None = None,
    alpha: float | None = None,
) -> tuple[int, int, float] | None:
    """Rebuild action LoRA before loading weights.

    Old checkpoints did not persist LoRA metadata.  Rank is recoverable from
    ``lora_A``.  Alpha is not encoded in a state dict, so the historical T-Rex
    default ``2 * rank`` is used only when neither CLI nor metadata provides it.
    Dropout is deliberately zero for inference (``eval`` would disable it
    anyway).
    """
    checkpoint_rank = infer_action_lora_rank(state_dict)
    if checkpoint_rank is None:
        return None

    saved_rank = training_args.get("action_lora_rank")
    resolved_rank = int(rank or saved_rank or checkpoint_rank)
    if resolved_rank != checkpoint_rank:
        raise ValueError(
            f"Action LoRA rank mismatch: requested {resolved_rank}, "
            f"checkpoint tensors encode {checkpoint_rank}"
        )
    saved_alpha = training_args.get("action_lora_alpha")
    resolved_alpha = float(
        alpha if alpha is not None else saved_alpha if saved_alpha is not None else 2 * resolved_rank
    )
    count = inject_action_expert_lora(
        model, rank=resolved_rank, alpha=resolved_alpha, dropout=0.0
    )
    expected = sum(key.endswith(".lora_A") for key in state_dict)
    if count != expected:
        raise RuntimeError(
            f"Reconstructed {count} action LoRA modules, but checkpoint contains "
            f"{expected} lora_A tensors"
        )
    return count, resolved_rank, resolved_alpha


def _feature_stats(
    stats: Mapping[str, Any], key: str, dimension: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block = stats.get(key)
    if not isinstance(block, Mapping):
        raise KeyError(f"Statistics are missing feature {key!r}")
    minimum = np.asarray(block.get("q01", block.get("min")), dtype=np.float32)
    maximum = np.asarray(block.get("q99", block.get("max")), dtype=np.float32)
    mask = np.asarray(block.get("mask", np.ones(dimension)), dtype=bool)
    if minimum.shape != (dimension,) or maximum.shape != (dimension,):
        raise ValueError(
            f"{key} statistics must be {dimension}-D, got "
            f"{minimum.shape}/{maximum.shape}"
        )
    if mask.shape != (dimension,):
        raise ValueError(f"{key} mask must be {dimension}-D, got {mask.shape}")
    return mask, minimum, maximum


def load_inference_statistics(
    path: str | Path,
    *,
    action_dim: int,
    use_robot_state: bool,
    dataset_name: str = "",
    tactile_dim: int = 60,
) -> dict[str, np.ndarray]:
    """Load either native LeRobot ``meta/stats.json`` or legacy T-Rex stats."""
    import json

    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)

    if "action" in raw and "observation.tactile" in raw:
        blocks = raw
        action_key = "action"
        state_key = "observation.state"
        tactile_key = "observation.tactile"
    else:
        if dataset_name:
            if dataset_name not in raw:
                raise KeyError(f"Dataset {dataset_name!r} is not present in {path}")
            blocks = raw[dataset_name]
        else:
            if not raw:
                raise ValueError(f"Statistics JSON is empty: {path}")
            blocks = raw[next(iter(raw))]
        action_key = "action"
        state_key = "state"
        tactile_key = "tactile_f6"

    action_mask, action_min, action_max = _feature_stats(
        blocks, action_key, action_dim
    )
    tactile_mask, tactile_min, tactile_max = _feature_stats(
        blocks, tactile_key, tactile_dim
    )
    result = {
        "action_mask": action_mask,
        "action_min": action_min,
        "action_max": action_max,
        "tacf6_mask": tactile_mask,
        "tacf6_min": tactile_min,
        "tacf6_max": tactile_max,
    }
    if use_robot_state:
        state_mask, state_min, state_max = _feature_stats(
            blocks, state_key, action_dim
        )
        result.update(
            state_mask=state_mask,
            state_min=state_min,
            state_max=state_max,
        )
    return result
