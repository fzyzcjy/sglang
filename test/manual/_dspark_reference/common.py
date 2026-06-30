# Upstream source: deepspec/modeling/dspark/common.py
# Copied verbatim except:
# - `from deepspec.utils.metrics import add_metric` replaced with a no-op shim
# - `from torch.nn.attention.flex_attention import create_block_mask` retained
#   (only used at runtime for training; parity tests do not call the attention
#   mask builder, so the import is guarded with a try/except for environments
#   that do not have flex_attention).
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

try:
    from torch.nn.attention.flex_attention import (
        create_block_mask as _create_block_mask,
    )

    _FLEX_ATTN_AVAILABLE = True
except ImportError:
    _create_block_mask = None
    _FLEX_ATTN_AVAILABLE = False


def _add_metric_noop(*args, **kwargs) -> None:
    pass


add_metric = _add_metric_noop


@dataclass
class DSparkForwardOutput:
    """Outputs for one DSpark training forward."""

    draft_logits: torch.Tensor
    target_ids: torch.Tensor
    eval_mask: torch.Tensor
    block_keep_mask: torch.Tensor
    confidence_pred: Optional[torch.Tensor] = None
    aligned_target_logits: Optional[torch.Tensor] = None


class AcceptRatePredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


def extract_context_feature(hidden_states, layer_ids):
    return torch.cat(
        [
            hidden_states[0 if layer_id == -1 else layer_id + 1]
            for layer_id in layer_ids
        ],
        dim=-1,
    )


def validate_target_layer_ids(layer_ids, num_target_layers: int):
    layer_ids = [int(layer_id) for layer_id in layer_ids]
    assert layer_ids, "target_layer_ids must not be empty."
    start = 0
    end = int(num_target_layers) - 1
    previous = None
    for layer_id in layer_ids:
        assert layer_id == -1 or start <= layer_id <= end, (
            f"target_layer_id {layer_id} is out of range {{-1}} U [{start}, {end}] "
            f"for num_target_layers={num_target_layers}. "
            "-1 denotes the embedding output."
        )
        assert (
            previous is None or layer_id > previous
        ), "target_layer_ids must be strictly increasing."
        previous = layer_id
    return layer_ids


def create_dspark_attention_mask(
    *,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
    device: torch.device,
):
    if not _FLEX_ATTN_AVAILABLE:
        raise RuntimeError(
            "create_dspark_attention_mask requires torch flex_attention (PyTorch >= 2.5)."
        )

    def dspark_mask_mod(b, h, q_idx, kv_idx):
        del h
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]
        is_context = kv_idx < seq_len
        mask_context = is_context & (kv_idx < anchor_pos)
        is_draft = kv_idx >= seq_len
        kv_block_id = (kv_idx - seq_len) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)
        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    bsz, num_blocks = anchor_positions.shape
    return _create_block_mask(
        dspark_mask_mod,
        B=bsz,
        H=None,
        Q_LEN=num_blocks * block_size,
        KV_LEN=seq_len + num_blocks * block_size,
        device=device,
    )


def build_anchor_candidate_mask(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    num_candidates = max(seq_len - 1, 0)
    if num_candidates == 0:
        return loss_mask[:, :0].bool()

    anchor_valid = loss_mask[:, :num_candidates] > 0.5
    first_target_valid = loss_mask[:, 1 : num_candidates + 1] > 0.5
    return anchor_valid & first_target_valid


def sample_anchor_positions(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    num_anchors: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = build_anchor_candidate_mask(
        seq_len=seq_len,
        loss_mask=loss_mask,
    )
    valid_counts = valid.sum(dim=1)
    bsz = loss_mask.shape[0]
    num_candidates = valid.shape[1]
    max_n = int(num_anchors)
    if num_candidates == 0:
        anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
        keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
        return anchors, keep_mask

    indices = (
        torch.arange(num_candidates, device=device)
        .unsqueeze(0)
        .expand(
            bsz,
            -1,
        )
    )
    masked_indices = torch.where(
        valid,
        indices,
        torch.full_like(indices, seq_len + 1),
    )
    random_vals = torch.rand(bsz, num_candidates, device=device)
    random_vals = torch.where(valid, random_vals, torch.full_like(random_vals, 2.0))
    _, sorted_idx = random_vals.sort(dim=1)
    gathered = torch.gather(masked_indices, 1, sorted_idx)
    if num_candidates < max_n:
        pad = torch.full(
            (bsz, max_n - num_candidates),
            seq_len + 1,
            dtype=gathered.dtype,
            device=device,
        )
        gathered = torch.cat([gathered, pad], dim=1)
    anchors = gathered[:, :max_n].sort(dim=1).values
    keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < (
        valid_counts.unsqueeze(1).clamp(max=max_n)
    )
    anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))
    return anchors, keep_mask


def build_eval_mask(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    label_indices: torch.Tensor,
    safe_label_indices: torch.Tensor,
    block_keep_mask: torch.Tensor,
) -> torch.Tensor:
    target_valid = label_indices < seq_len
    target_loss_mask = torch.gather(
        loss_mask.unsqueeze(1).expand(-1, label_indices.size(1), -1),
        2,
        safe_label_indices,
    )
    eval_mask = target_valid & (target_loss_mask > 0.5)
    eval_mask = eval_mask & block_keep_mask.unsqueeze(-1)
    return eval_mask.to(torch.int32).cumprod(dim=-1).bool()


@torch.no_grad()
def log_sampler_stats(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    block_keep_mask: torch.Tensor,
    eval_mask: torch.Tensor,
    block_size: int,
    num_anchors: int,
) -> None:
    pass


def create_position_ids(
    anchor_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    bsz, num_blocks = anchor_positions.shape
    device = anchor_positions.device
    offsets = torch.arange(block_size, device=device).view(1, 1, -1)
    return (anchor_positions.unsqueeze(-1) + offsets).view(
        bsz,
        num_blocks * block_size,
    )


def create_noise_embed(
    embed_tokens: nn.Module,
    input_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    mask_token_id: int,
    block_size: int,
) -> torch.Tensor:
    bsz = input_ids.shape[0]
    num_blocks = anchor_positions.shape[1]
    device = input_ids.device
    noise_ids = torch.full(
        (bsz, num_blocks * block_size),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    block_starts = torch.arange(num_blocks, device=device) * block_size
    block_starts = block_starts.unsqueeze(0).expand(bsz, -1)
    anchor_tokens = torch.gather(input_ids, 1, anchor_positions)
    flat_batch_idx = (
        torch.arange(bsz, device=device)
        .unsqueeze(1)
        .expand(
            bsz,
            num_blocks,
        )
    )
    noise_ids[flat_batch_idx, block_starts] = torch.where(
        block_keep_mask,
        anchor_tokens,
        torch.tensor(mask_token_id, dtype=torch.long, device=device),
    )
    return embed_tokens(noise_ids)


__all__ = [
    "DSparkForwardOutput",
    "AcceptRatePredictor",
    "extract_context_feature",
    "validate_target_layer_ids",
    "create_dspark_attention_mask",
    "build_anchor_candidate_mask",
    "sample_anchor_positions",
    "build_eval_mask",
    "log_sampler_stats",
    "create_position_ids",
    "create_noise_embed",
]
