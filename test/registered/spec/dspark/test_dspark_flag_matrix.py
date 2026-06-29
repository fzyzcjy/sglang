import importlib.util
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-large")

DEFAULT_TARGET_MODEL_DSPARK_QWEN3 = "Qwen/Qwen3-8B"
DEFAULT_DRAFT_MODEL_DSPARK_QWEN3 = "z-lab/Qwen3-8B-DSpark"

DEFAULT_TARGET_MODEL_DSPARK_GEMMA4 = "google/gemma-4-9b"
DEFAULT_DRAFT_MODEL_DSPARK_GEMMA4 = "z-lab/Gemma4-9B-DSpark"

# The ragged-verify flag (SGLANG_RAGGED_VERIFY) and RaggedVerifyLayout are owned
# by the ragged-verify-infra chapter and are not yet landed in any speculative
# tree. cutoff-only and full matrix entries depend on them, so they are gated.
_RAGGED_VERIFY_AVAILABLE = (
    importlib.util.find_spec("sglang.srt.speculative.ragged_verify") is not None
)

_MATRIX_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "The three primary colors are red, green, and",
]


def _greedy_request(url: str, prompt: str, max_new_tokens: int = 48) -> str:
    """Send a greedy generation request and return the output text."""
    resp = requests.post(
        url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        },
    )
    resp.raise_for_status()
    return resp.json()["text"]


class _DSparkFlagMatrixBase(CustomTestCase):
    """Lossless flag matrix (off / cutoff-only / full) for one model pair.

    `off` (static full block) is the baseline. `cutoff-only` must be bit-equal
    to `off` when given full gamma. `full` (real-N ragged) must equal
    `cutoff-only` under the same n-2-frozen ell_r. The server's output-text
    losslessness is asserted here; the backend-level `ragged == concat-of-uniform`
    bitwise check is owned by the dense / dsv4 backend chapters and only
    referenced (not re-implemented) here.
    """

    target_model: str = ""
    draft_model: str = ""
    attention_backend: str = "flashinfer"
    mem_fraction_static: float = 0.7
    disable_overlap: bool = True

    @classmethod
    def _base_args(cls) -> list:
        return [
            "--trust-remote-code",
            "--attention-backend",
            cls.attention_backend,
            "--mem-fraction-static",
            str(cls.mem_fraction_static),
            "--page-size",
            "1",
        ]

    @classmethod
    def _spec_args(cls) -> list:
        args = cls._base_args() + [
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            cls.draft_model,
        ]
        if cls.disable_overlap:
            args.append("--disable-overlap-schedule")
        return args

    def _maybe_skip_models(self) -> None:
        if not self.target_model or not self.draft_model:
            self.skipTest(
                "Model paths not configured. Set target_model and draft_model."
            )

    def _launch_and_capture(self, extra_env: dict | None = None) -> dict:
        """Launch a DSpark spec server, capture greedy outputs, tear down."""
        base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        proc = popen_launch_server(
            self.target_model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self._spec_args(),
            env=env,
        )
        try:
            return {p: _greedy_request(base_url, p) for p in _MATRIX_PROMPTS}
        finally:
            kill_process_tree(proc.pid)

    def test_off_mode_greedy_baseline_runs(self):
        """`off` (static full block) greedy outputs are captured as the baseline."""
        self._maybe_skip_models()
        off_outputs = self._launch_and_capture()
        for prompt in _MATRIX_PROMPTS:
            self.assertIn(prompt, off_outputs)
            self.assertIsInstance(off_outputs[prompt], str)

    def test_cutoff_only_bit_equal_to_off_with_full_gamma(self):
        """`cutoff-only` with full gamma must be bit-equal to `off` greedy output."""
        self._maybe_skip_models()
        if not _RAGGED_VERIFY_AVAILABLE:
            self.skipTest(
                "RAGGED_VERIFY not yet implemented (cutoff-only flag fixture "
                "blocked on ragged-verify-infra: SGLANG_RAGGED_VERIFY / "
                "RaggedVerifyLayout)."
            )
        off_outputs = self._launch_and_capture()
        # cutoff-only with full gamma == off (no suffix truncated). The exact env
        # wiring is owned by ragged-verify-infra; this asserts the contract.
        cutoff_env = {"SGLANG_RAGGED_VERIFY": "cutoff_only"}
        cutoff_outputs = self._launch_and_capture(extra_env=cutoff_env)
        for prompt in _MATRIX_PROMPTS:
            self.assertEqual(
                cutoff_outputs[prompt],
                off_outputs[prompt],
                f"cutoff-only (full gamma) != off for prompt {prompt!r}",
            )

    @unittest.skip(
        "BLOCKED on ragged-verify routing decision: the `full` real-N path "
        "(ragged-verify execution + num-tokens-keyed cuda-graph) is under team "
        "design discussion. This e2e entry is stubbed and not wired to run."
    )
    def test_full_mode_equals_cutoff_only_same_frozen_ell(self):
        """`full` (real-N ragged) must equal `cutoff-only` under same frozen ell_r."""
        self._maybe_skip_models()
        off_outputs = self._launch_and_capture()
        cutoff_env = {"SGLANG_RAGGED_VERIFY": "cutoff_only"}
        cutoff_outputs = self._launch_and_capture(extra_env=cutoff_env)
        full_env = {"SGLANG_RAGGED_VERIFY": "full"}
        full_outputs = self._launch_and_capture(extra_env=full_env)
        for prompt in _MATRIX_PROMPTS:
            self.assertEqual(
                full_outputs[prompt],
                cutoff_outputs[prompt],
                f"full != cutoff-only for prompt {prompt!r}",
            )
            self.assertEqual(
                full_outputs[prompt],
                off_outputs[prompt],
                f"full != off for prompt {prompt!r}",
            )


class TestDSparkFlagMatrixQwen3(_DSparkFlagMatrixBase):
    """DSpark off/cutoff-only/full lossless matrix for Qwen3 (no overlap)."""

    target_model = DEFAULT_TARGET_MODEL_DSPARK_QWEN3
    draft_model = DEFAULT_DRAFT_MODEL_DSPARK_QWEN3
    disable_overlap = True


class TestDSparkFlagMatrixGemma4(_DSparkFlagMatrixBase):
    """DSpark off/cutoff-only/full lossless matrix for Gemma4 (no overlap)."""

    target_model = DEFAULT_TARGET_MODEL_DSPARK_GEMMA4
    draft_model = DEFAULT_DRAFT_MODEL_DSPARK_GEMMA4
    disable_overlap = True


@unittest.skip(
    "BLOCKED on dsv4 base + spike: dsv4 draft model id is unnamed and the dsv4 "
    "backend is not yet landed. dsv4 e2e flag matrix is a stub."
)
class TestDSparkFlagMatrixDsv4(_DSparkFlagMatrixBase):
    """DSpark dsv4 flag matrix stub (blocked on dsv4 base + draft model id)."""

    target_model = ""
    draft_model = ""


if __name__ == "__main__":
    unittest.main()
