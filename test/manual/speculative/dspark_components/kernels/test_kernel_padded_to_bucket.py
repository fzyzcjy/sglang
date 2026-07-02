import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.padded_to_bucket import (
    pad_verify_lens_to_bucket,
    pad_verify_lens_to_bucket_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)


@pytest.mark.parametrize(
    "bs,padded_bs",
    [(1, 1), (2, 2), (3, 3), (2, 4), (3, 16), (8, 128), (1, 64), (5, 5)],
)
def test_triton_matches_torch_eager_and_padded_buckets(bs, padded_bs):
    device = torch.device("cuda")
    num_draft = 6
    graph_num_tokens = padded_bs * num_draft
    verify_lens = torch.randint(
        1, num_draft + 1, (bs,), dtype=torch.int32, device=device
    )
    ref = pad_verify_lens_to_bucket(
        verify_lens=verify_lens,
        graph_num_tokens=graph_num_tokens,
        bs=bs,
        num_draft_tokens=num_draft,
    )
    got = pad_verify_lens_to_bucket_triton(
        verify_lens=verify_lens,
        graph_num_tokens=graph_num_tokens,
        bs=bs,
        num_draft_tokens=num_draft,
    )
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)
