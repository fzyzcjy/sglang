import unittest
from types import SimpleNamespace
from typing import Optional

from sglang.srt.speculative.draft_worker_common import (
    _DEEPSEEK_V4_DRAFT_BACKEND,
    _resolve_draft_attention_backend_fallback,
    _select_draft_attention_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_draft_server_args(
    *, draft_backend: Optional[str], inferred_backend: Optional[str] = None
) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_draft_attention_backend=draft_backend,
        get_attention_backends=lambda: (inferred_backend, None),
    )


class TestDSparkDraftBackendSelection(CustomTestCase):
    def test_deepseek_v4_draft_config_selects_v4_sparse_backend(self):
        """A DeepSeek-V4 DSpark draft config selects the V4 sparse draft backend."""
        draft_hf_config = SimpleNamespace(architectures=["DeepseekV4ForCausalLMDSpark"])
        draft_server_args = _make_draft_server_args(draft_backend="flashinfer")

        resolved = _select_draft_attention_backend(
            draft_hf_config=draft_hf_config,
            draft_server_args=draft_server_args,
            algo_label="DSPARK",
        )

        self.assertEqual(resolved, _DEEPSEEK_V4_DRAFT_BACKEND)
        self.assertEqual(resolved, "dsv4")

    def test_non_v4_draft_config_keeps_supported_backend_unchanged(self):
        """A non-V4 draft config keeps the user's supported backend unchanged."""
        draft_hf_config = SimpleNamespace(architectures=["LlamaForCausalLM"])
        draft_server_args = _make_draft_server_args(draft_backend="fa3")

        resolved = _select_draft_attention_backend(
            draft_hf_config=draft_hf_config,
            draft_server_args=draft_server_args,
            algo_label="DFLASH",
        )

        self.assertEqual(resolved, "fa3")

    def test_missing_draft_config_falls_back_unchanged(self):
        """A missing draft config falls back to the existing allow-list resolution."""
        draft_server_args = _make_draft_server_args(draft_backend="triton")

        resolved = _select_draft_attention_backend(
            draft_hf_config=None,
            draft_server_args=draft_server_args,
            algo_label="DSPARK",
        )

        self.assertEqual(resolved, "triton")

    def test_non_v4_unknown_backend_downgrades_like_today(self):
        """A non-V4 draft config with an unknown backend downgrades exactly as before."""
        draft_hf_config = SimpleNamespace(architectures=["LlamaForCausalLM"])
        draft_server_args = _make_draft_server_args(draft_backend="some_unknown")

        selected = _select_draft_attention_backend(
            draft_hf_config=draft_hf_config,
            draft_server_args=draft_server_args,
            algo_label="DFLASH",
        )
        fallback = _resolve_draft_attention_backend_fallback(
            draft_server_args=draft_server_args, algo_label="DFLASH"
        )

        self.assertEqual(selected, fallback)
        self.assertIn(fallback, ("flashinfer", "triton"))

    def test_non_v4_no_backend_infers_from_server_args(self):
        """A non-V4 draft config with no explicit backend uses the inferred backend."""
        draft_hf_config = SimpleNamespace(architectures=["Qwen2ForCausalLM"])
        draft_server_args = _make_draft_server_args(
            draft_backend=None, inferred_backend="fa4"
        )

        resolved = _select_draft_attention_backend(
            draft_hf_config=draft_hf_config,
            draft_server_args=draft_server_args,
            algo_label="DFLASH",
        )

        self.assertEqual(resolved, "fa4")


if __name__ == "__main__":
    unittest.main()
