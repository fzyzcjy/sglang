import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.speculative_hook import (
    _handle_dspark,
    _target_checkpoint_bundles_dspark_draft,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_BUNDLED_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
_PLAIN_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash"


def _bundled_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_markov_rank=256,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_noise_token_id=128799,
    )


def _plain_hf_config() -> SimpleNamespace:
    return SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])


def _make_dspark_server_args(*, model_path: str, hf_config: SimpleNamespace) -> ServerArgs:
    """Build a DSpark ServerArgs stub with an injected model config (skips __post_init__).

    ``model_path="dummy"`` short-circuits ServerArgs.__post_init__, then the fields
    ``_handle_dspark`` reads are set explicitly and a fake model config is injected so
    ``get_model_config()`` returns it without touching the network. ``dspark_block_size``
    is preset so ``_handle_dspark`` never falls into the get_config gamma inference.
    """
    server_args = ServerArgs(model_path="dummy")
    server_args.model_path = model_path
    server_args.device = "cuda"
    server_args.speculative_algorithm = "DSPARK"
    server_args.speculative_draft_model_path = None
    server_args.speculative_dspark_block_size = 5
    server_args.model_config = SimpleNamespace(hf_config=hf_config)
    return server_args


class TestTargetCheckpointBundlesDsparkDraft(CustomTestCase):
    def test_bundled_dsv4_config_is_detected(self):
        """A config carrying dspark_* fields is recognized as a bundled draft."""
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        self.assertTrue(_target_checkpoint_bundles_dspark_draft(server_args))

    def test_plain_target_config_is_not_detected(self):
        """A plain target config with no dspark_* fields is not a bundled draft."""
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        self.assertFalse(_target_checkpoint_bundles_dspark_draft(server_args))


class TestDsparkDraftPathDefaulting(CustomTestCase):
    def test_bundled_checkpoint_defaults_draft_path_to_model_path(self):
        """DSpark with a bundled checkpoint and no draft path reuses the model path."""
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        _handle_dspark(server_args)
        self.assertEqual(server_args.speculative_draft_model_path, _BUNDLED_MODEL_PATH)
        self.assertEqual(server_args.speculative_num_draft_tokens, 6)

    def test_plain_target_without_draft_path_raises(self):
        """DSpark with a plain target and no draft path still requires an explicit path."""
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        with self.assertRaises(ValueError):
            _handle_dspark(server_args)

    def test_explicit_draft_path_is_not_overwritten(self):
        """An explicit --speculative-draft-model-path is preserved, not defaulted away."""
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.speculative_draft_model_path = "deepseek-ai/some-other-dspark-draft"
        _handle_dspark(server_args)
        self.assertEqual(
            server_args.speculative_draft_model_path,
            "deepseek-ai/some-other-dspark-draft",
        )


if __name__ == "__main__":
    unittest.main()
