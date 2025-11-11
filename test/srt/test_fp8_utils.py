import torch
import unittest

from sglang.test.test_utils import CustomTestCase

from python.sglang.srt.layers.quantization.fp8_utils import transform_scale_ue8m0, inverse_transform_scale_ue8m0


class TestInverseTransformScaleUe8m0(CustomTestCase):
    def test_round_trip(self):
        sf_fp32 = TODO
        mn = TODO

        sf_packed = transform_scale_ue8m0(sf_fp32, mn=mn)
        sf_fp32 = inverse_transform_scale_ue8m0(sf_packed, mn=mn)

        sf_packed_recreated = transform_scale_ue8m0(sf_fp32, mn=mn)
        assert torch.all(sf_packed == sf_packed_recreated), f"{sf_packed=} {sf_packed_recreated}"


if __name__ == "__main__":
    unittest.main()
