from typing import Iterable, List

import torch


def expert_model_parallel_shuffle_inplace(
    origin_global_experts_indices: torch.Tensor,  # [num_layers, total_num_experts]
    experts_weight: List[
        Iterable[torch.Tensor]
    ],  # num_layers * n * [num_local_experts, hidden_size_i]
    current_global_experts_indices: torch.Tensor,  # [num_layers, total_num_experts]
    ep: torch.distributed.ProcessGroup,
) -> None:
    num_layers = origin_global_experts_indices.size(0)
    total_num_experts = origin_global_experts_indices.size(1)
    assert len(experts_weight) == num_layers
    num_local_experts = experts_weight[0][0].size(0)
    assert current_global_experts_indices.size() == (num_layers, total_num_experts)

    ep_rank = ep.rank()
    assert total_num_experts == ep.size() * num_local_experts

    current_experts_weight = [torch.empty_like(weight) for weight in experts_weight[0]]

    def _shuffle_layer(
        origin_global_experts_indices: torch.Tensor,  # [total_num_experts]
        experts_weight: Iterable[
            torch.Tensor
        ],  # n * [num_local_experts, hidden_size_i]
        current_global_experts_indices: torch.Tensor,  # [total_num_experts]
    ):
        nonlocal total_num_experts
        nonlocal num_local_experts
        nonlocal ep
        nonlocal ep_rank
        nonlocal current_experts_weight

        origin_global_experts_indices = origin_global_experts_indices.tolist()
        current_global_experts_indices = current_global_experts_indices.tolist()

        # 没有变的 expert 就不用 copy 了
        dst_received = [
            origin_global_experts_indices[ep_rank * num_local_experts + i]
            == current_global_experts_indices[ep_rank * num_local_experts + i]
            for i in range(num_local_experts)
        ]

        # 先做卡内 origin->current 的 copy
        for src in range(num_local_experts):
            for dst in range(num_local_experts):
                if not dst_received[dst] and (
                    origin_global_experts_indices[ep_rank * num_local_experts + src]
                    == current_global_experts_indices[ep_rank * num_local_experts + dst]
                ):
                    dst_received[dst] = True
                    for i in range(len(experts_weight)):
                        current_experts_weight[i][dst].copy_(experts_weight[i][src])

        p2p_ops = []

        def _find_all_ep_rank_with_expert(expert):
            nonlocal origin_global_experts_indices
            nonlocal current_global_experts_indices
            nonlocal num_local_experts

            ep_rank_to_send = []
            ep_rank_to_recv = []
            for i, e in enumerate(origin_global_experts_indices):
                if e == expert:
                    rank = i // num_local_experts
                    if not ep_rank_to_send or ep_rank_to_send[-1] != rank:
                        ep_rank_to_send.append(rank)
            for i, e in enumerate(current_global_experts_indices):
                if e == expert:
                    rank = i // num_local_experts
                    if not ep_rank_to_recv or ep_rank_to_recv[-1] != rank:
                        ep_rank_to_recv.append(rank)
            # 减去卡内 copy 过的
            result = []
            i, j = 0, 0
            while i < len(ep_rank_to_send) and j < len(ep_rank_to_recv):
                if ep_rank_to_send[i] < ep_rank_to_recv[j]:
                    i += 1
                elif ep_rank_to_send[i] > ep_rank_to_recv[j]:
                    result.append(ep_rank_to_recv[j])
                    j += 1
                else:
                    i += 1
                    j += 1
            result.extend(ep_rank_to_recv[j:])
            return ep_rank_to_send, result

        # 处理要发送的 experts
        experts_sent = set()
        experts = []
        for src in range(num_local_experts):
            expert = origin_global_experts_indices[ep_rank * num_local_experts + src]
            if expert in experts_sent:
                continue
            experts_sent.add(expert)
            experts.append((expert, src))

        for expert, src in sorted(experts):
            ep_rank_to_send, ep_rank_to_recv = _find_all_ep_rank_with_expert(expert)

            # 跨卡 p2p，TODO(tyx): 这里可以考虑优先机器内通信和 broadcast
            num_dst_per_src = len(ep_rank_to_recv) // len(ep_rank_to_send)
            i = ep_rank_to_send.index(ep_rank)
            for dst in ep_rank_to_recv[i * num_dst_per_src : (i + 1) * num_dst_per_src]:
                p2p_ops.extend(
                    [
                        torch.distributed.P2POp(
                            torch.distributed.isend,
                            weight[src],
                            torch.distributed.get_global_rank(ep, dst),
                        )
                        for weight in experts_weight
                    ]
                )
            if i + num_dst_per_src * len(ep_rank_to_send) < len(ep_rank_to_recv):
                dst = ep_rank_to_recv[i + num_dst_per_src * len(ep_rank_to_send)]
                p2p_ops.extend(
                    [
                        torch.distributed.P2POp(
                            torch.distributed.isend,
                            weight[src],
                            torch.distributed.get_global_rank(ep, dst),
                        )
                        for weight in experts_weight
                    ]
                )

        # 处理要接收的 experts
        experts_receive_idx = {}
        experts = []
        for dst in range(num_local_experts):
            if dst_received[dst]:
                continue
            expert = current_global_experts_indices[ep_rank * num_local_experts + dst]
            if expert in experts_receive_idx:
                continue
            experts_receive_idx[expert] = dst
            experts.append((expert, dst))

        for expert, dst in sorted(experts):
            ep_rank_to_send, ep_rank_to_recv = _find_all_ep_rank_with_expert(expert)

            # 跨卡 p2p，TODO(tyx): 这里可以考虑优先机器内通信和 broadcast
            num_dst_per_src = len(ep_rank_to_recv) // len(ep_rank_to_send)
            j = ep_rank_to_recv.index(ep_rank)
            if j < len(ep_rank_to_send) * num_dst_per_src:
                src = ep_rank_to_send[j // num_dst_per_src]
            else:
                src = ep_rank_to_send[j - len(ep_rank_to_send) * num_dst_per_src]

            p2p_ops.extend(
                [
                    torch.distributed.P2POp(
                        torch.distributed.irecv,
                        weight[dst],
                        torch.distributed.get_global_rank(ep, src),
                    )
                    for weight in current_experts_weight
                ]
            )

        if p2p_ops:
            reqs = torch.distributed.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

        # 做卡内 current 的 copy
        for i in range(num_local_experts):
            if (
                origin_global_experts_indices[ep_rank * num_local_experts + i]
                == current_global_experts_indices[ep_rank * num_local_experts + i]
            ):
                continue
            if dst_received[i]:
                for j in range(len(experts_weight)):
                    experts_weight[j][i].copy_(current_experts_weight[j][i])
            else:
                expert = current_global_experts_indices[ep_rank * num_local_experts + i]
                src = experts_receive_idx[expert]
                for j in range(len(experts_weight)):
                    experts_weight[j][i].copy_(current_experts_weight[j][src])

    for idx in range(num_layers):
        _shuffle_layer(
            origin_global_experts_indices[idx],
            experts_weight[idx],
            current_global_experts_indices[idx],
        )
