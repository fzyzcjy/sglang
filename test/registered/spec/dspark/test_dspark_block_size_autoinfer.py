import unittest
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest import mock

from sglang.srt.arg_groups.speculative_hook import _handle_dspark
from sglang.srt.speculative.dspark_utils import DEFAULT_DSPARK_GAMMA
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_server_args(
    *,
    speculative_dspark_block_size: Optional[int] = None,
    speculative_num_draft_tokens: Optional[int] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        enable_dp_attention=False,
        pp_size=1,
        speculative_draft_model_path="/fake/dspark/draft",
        speculative_draft_model_revision=None,
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_dspark_block_size=speculative_dspark_block_size,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        json_model_override_args="{}",
        trust_remote_code=False,
        max_running_requests=None,
        enable_mixed_chunk=False,
    )


def _make_draft_config(*, block_size: Optional[int]) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "num_hidden_layers": 4,
        "num_target_layers": 28,
        "target_layer_ids": [6, 13, 20, 27],
        "markov_rank": 32,
        "markov_head_type": "vanilla",
        "mask_token_id": 1,
    }
    if block_size is not None:
        config["block_size"] = block_size
    return config


class TestDSparkBlockSizeAutoInfer(CustomTestCase):
    def test_auto_infer_sets_num_draft_tokens_to_block_size_plus_one(self):
        """With no flags, num_draft_tokens is inferred as draft block_size + 1."""
        server_args = _make_server_args()
        with mock.patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=_make_draft_config(block_size=5),
        ):
            _handle_dspark(server_args)
        self.assertEqual(server_args.speculative_num_draft_tokens, 6)

    def test_falls_back_to_default_gamma_when_block_size_missing(self):
        """With no flags and no draft block_size, num_draft_tokens falls back to DEFAULT_DSPARK_GAMMA + 1."""
        server_args = _make_server_args()
        with mock.patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=_make_draft_config(block_size=None),
        ):
            _handle_dspark(server_args)
        self.assertEqual(
            server_args.speculative_num_draft_tokens, DEFAULT_DSPARK_GAMMA + 1
        )

    def test_consistency_check_raises_on_mismatching_num_draft_tokens(self):
        """Only --speculative-num-draft-tokens given must equal draft block_size + 1, else ValueError."""
        server_args = _make_server_args(speculative_num_draft_tokens=10)
        with mock.patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=_make_draft_config(block_size=5),
        ):
            with self.assertRaises(ValueError):
                _handle_dspark(server_args)

    def test_consistency_check_passes_on_matching_num_draft_tokens(self):
        """Only --speculative-num-draft-tokens given that matches block_size + 1 is accepted."""
        server_args = _make_server_args(speculative_num_draft_tokens=6)
        with mock.patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=_make_draft_config(block_size=5),
        ):
            _handle_dspark(server_args)
        self.assertEqual(server_args.speculative_num_draft_tokens, 6)

    def test_consistency_check_skipped_when_block_size_missing(self):
        """A draft config without block_size skips the consistency check gracefully."""
        server_args = _make_server_args(speculative_num_draft_tokens=10)
        with mock.patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=_make_draft_config(block_size=None),
        ):
            _handle_dspark(server_args)
        self.assertEqual(server_args.speculative_num_draft_tokens, 10)

    def test_explicit_block_size_sets_num_draft_tokens(self):
        """An explicit --speculative-dspark-block-size sets num_draft_tokens to gamma + 1 without reading the config."""
        server_args = _make_server_args(speculative_dspark_block_size=7)
        with mock.patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            side_effect=AssertionError("get_config must not be called"),
        ):
            _handle_dspark(server_args)
        self.assertEqual(server_args.speculative_num_draft_tokens, 8)


if __name__ == "__main__":
    unittest.main()
