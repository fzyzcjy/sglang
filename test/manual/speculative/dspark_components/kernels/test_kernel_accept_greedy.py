import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.accept_greedy import (
    accept_greedy,
    accept_greedy_triton,
    gather_row_bonus_triton,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)

T = 6
V = 200


@pytest.mark.parametrize("bs", [1, 2, 3, 8])
@pytest.mark.parametrize("with_cutoff", [True, False])
def test_accept_greedy_triton_matches_torch(bs, with_cutoff):
    device = torch.device("cuda")
    candidates = torch.randint(0, V, (bs, T), dtype=torch.int64, device=device)
    target_logits = torch.randn(bs * T, V, device=device)
    cutoff = None
    if with_cutoff:
        verify_lens = torch.randint(1, T + 1, (bs,), dtype=torch.int32, device=device)
        cutoff = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=verify_lens, graph_num_tokens=bs * T
        )
    cl_r, b_r, tr_r = accept_greedy(
        candidates=candidates,
        target_logits=target_logits,
        verify_num_draft_tokens=T,
        cutoff_layout=cutoff,
    )
    cl_g, b_g, tr_g = accept_greedy_triton(
        candidates=candidates,
        target_logits=target_logits,
        verify_num_draft_tokens=T,
        cutoff_layout=cutoff,
    )
    assert torch.equal(cl_g, cl_r)
    assert torch.equal(b_g, b_r)
    assert torch.equal(tr_g, tr_r)


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 64])
def test_gather_row_bonus_matches_torch(bs):
    device = torch.device("cuda")
    cols = T
    table = torch.randint(0, 129280, (bs, cols), dtype=torch.int64, device=device)
    idx = torch.randint(0, cols, (bs,), dtype=torch.int32, device=device)
    ref = table[torch.arange(bs, device=device), idx.long()].to(torch.int64)
    got = gather_row_bonus_triton(table=table, idx=idx)
    assert torch.equal(got, ref)
