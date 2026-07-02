import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.qo_indptr import (
    build_qo_indptr,
    build_qo_indptr_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 64, 129])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_triton_matches_torch_indptr(bs, dtype):
    """triton qo_indptr/extend_start_loc are bit-exact vs the torch cumsum/cat/clone chain."""
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(bs)
    verify_lens = torch.randint(
        1, 8, (bs,), device=device, generator=g, dtype=dtype
    )

    ref = build_qo_indptr(verify_lens=verify_lens)
    got = build_qo_indptr_triton(verify_lens=verify_lens.to(torch.int32))

    assert torch.equal(got.qo_indptr, ref.qo_indptr)
    assert torch.equal(got.extend_start_loc, ref.extend_start_loc)
    assert got.qo_indptr.dtype == torch.int32
    assert got.extend_start_loc.dtype == torch.int32


def test_outputs_are_distinct_storage():
    """extend_start_loc must not alias qo_indptr (consumers copy_ into bs-axis buffers)."""
    device = torch.device("cuda")
    verify_lens = torch.tensor([3, 1, 5], device=device, dtype=torch.int32)
    got = build_qo_indptr_triton(verify_lens=verify_lens)
    got.extend_start_loc.fill_(-7)
    assert int(got.qo_indptr[0]) == 0
    assert int(got.qo_indptr[1]) == 3
