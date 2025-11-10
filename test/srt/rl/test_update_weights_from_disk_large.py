import unittest

from test_update_weights_from_disk import TestServerUpdateWeightsFromDisk


class TestServerUpdateWeightsFromDiskDeepSeek(TestServerUpdateWeightsFromDisk):
    model = "deepseek-ai/DeepSeek-V3-0324"


if __name__ == "__main__":
    unittest.main()
