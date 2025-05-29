from typing import Optional, Literal

import torch
from sglang.srt.managers.eplb_algorithms import deepseek_vec, deepseek

EplbAlgorithm = Literal['TODO']


def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: Optional[int],
    num_nodes: int,
    algorithm: EplbAlgorithm,
):
    if TODO:
        return deepseek_vec.rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
        )

    if TODO:
        return deepseek.rebalance_experts(

        )
