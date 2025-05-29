from typing import Optional, Literal

import torch
from sglang.srt.managers.eplb_algorithms import deepseek_vec, deepseek

EplbAlgorithm = Literal[
    'deepseek',
    'deepseek_hierarchical',
    'deepseek_vec',
    'deepseek_vec_hierarchical',
]


def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: Optional[int],
    num_nodes: int,
    algorithm: EplbAlgorithm,
):
    if algorithm in ['deepseek_vec', 'deepseek_vec_hierarchical']:
        return deepseek_vec.rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            enable_hierarchical=algorithm == 'deepseek_vec_hierarchical',
        )

    if algorithm in ['deepseek', 'deepseek_hierarchical']:
        return deepseek.rebalance_experts(
            weight=tokens_per_expert.sum(dim=0),
            num_replicas=num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_physical_experts // num_local_physical_experts,
            enable_hierarchical=algorithm == 'deepseek_hierarchical',
        )

    raise NotImplementedError
