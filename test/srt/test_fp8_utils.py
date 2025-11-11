import torch
import unittest

from sglang.test.test_utils import CustomTestCase

from sglang.srt.layers.quantization.fp8_utils import transform_scale_ue8m0, inverse_transform_scale_ue8m0, \
    quant_weight_ue8m0


class TestInverseTransformScaleUe8m0(CustomTestCase):
    def test_round_trip(self):
        for _ in range(100):
            weight_bf16 = torch.randn(
                # DeepSeek V3 kv_b_proj
                (32768, 512),
                dtype=torch.bfloat16,
                device="cuda",
            )

            weight_block_size = [128, 128]

            qweight, sf_fp32 = quant_weight_ue8m0(weight_bf16, weight_block_size=weight_block_size)
            mn = qweight.shape[-2]

            sf_packed = transform_scale_ue8m0(sf_fp32, mn=mn)
            sf_fp32 = inverse_transform_scale_ue8m0(sf_packed, mn=mn)

            sf_packed_recreated = transform_scale_ue8m0(sf_fp32, mn=mn)
            assert torch.all(sf_packed == sf_packed_recreated), f"{sf_packed=} {sf_packed_recreated}"


if __name__ == "__main__":
    unittest.main()
