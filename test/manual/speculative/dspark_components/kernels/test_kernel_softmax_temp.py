import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.softmax_temp import (
    softmax_temp,
    softmax_temp_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)


@pytest.mark.parametrize("bs", [1, 2, 3, 8])
@pytest.mark.parametrize("rows_per_request", [1, 6, 7])
@pytest.mark.parametrize("vocab", [1000, 4096, 129280])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_triton_matches_torch_probs(bs, rows_per_request, vocab, dtype):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(bs * 1000 + rows_per_request)
    logits = (
        torch.randn(bs * rows_per_request, vocab, device=device, generator=g) * 8.0
    ).to(dtype)
    temperatures = (torch.rand(bs, device=device, generator=g) * 1.5 + 0.05).to(
        torch.float32
    )

    ref = softmax_temp(
        logits=logits, temperatures=temperatures, rows_per_request=rows_per_request
    )
    got = softmax_temp_triton(
        logits=logits, temperatures=temperatures, rows_per_request=rows_per_request
    )

    assert got.dtype == ref.dtype == torch.float32
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(
        got.sum(dim=-1), torch.ones_like(got.sum(dim=-1)), rtol=1e-5, atol=1e-5
    )


def test_column_temperatures_accepted():
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(7)
    logits = torch.randn(6, 512, device=device, generator=g).to(torch.bfloat16)
    temperatures = (torch.rand(2, 1, device=device, generator=g) + 0.3).to(
        torch.float32
    )
    ref = softmax_temp(logits=logits, temperatures=temperatures, rows_per_request=3)
    got = softmax_temp_triton(
        logits=logits, temperatures=temperatures, rows_per_request=3
    )
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-7)
