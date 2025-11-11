import unittest

from test_update_weights_from_disk import TestServerUpdateWeightsFromDisk as _ParentTest


class TestServerUpdateWeightsFromDiskDeepSeek(_ParentTest):
    model = "deepseek-ai/DeepSeek-V3-0324"
    model_after_update = "deepseek-ai/DeepSeek-V3"
    launch_server_other_args = ["--tp", "4"]


if __name__ == "__main__":
    unittest.main()
