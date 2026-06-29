"""DSpark tp_size=2 losslessness: cross-rank determinism + accuracy floor.

A tp=2 run must stay cross-rank consistent (every rank agrees, so the server is
deterministic within a run) and preserve the target model's output quality under
lossless greedy decoding.

It must NOT, however, be asserted token-for-token against the tp=1 baseline:
tensor parallelism splits each reduction across ranks, so the all-reduce sums
floating-point partials in a different order than a single GPU. That reorder
changes the low bits of every logit, which flips the argmax at any near-tie (top-2
logit gap <~0.25 nats) from the very first decode step. This is intrinsic to TP
and independent of DSpark -- a plain NON-spec tp=2 server already diverges
token-for-token from a non-spec tp=1 server on the first tokens. So tp=2-vs-tp=1
token equality is not a sound losslessness invariant for any model.

The sound invariants here, mirroring EAGLE/DFlash:
  * cross-rank determinism: two identical greedy requests to the tp=2 server
    return identical text (the ranks stay in lockstep within a run);
  * accuracy floor: GSM8K score and speculative accept length on the tp=2 server
    clear a comfortable threshold, proving the lossless accept path is intact and
    speculation is actually accepting drafts.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="base-b", runner_config="2-gpu-large")

DEFAULT_TARGET_MODEL_DSPARK_QWEN3 = "Qwen/Qwen3-4B"
DEFAULT_DRAFT_MODEL_DSPARK_QWEN3 = "deepseek-ai/dspark_qwen3_4b_block7"


def _greedy_request(url: str, prompt: str, max_new_tokens: int = 32) -> str:
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


def _checkpoints_available(*model_paths: str) -> bool:
    """Probe whether every HF repo path resolves (False if gated/missing/offline)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return True

    api = HfApi()
    for path in model_paths:
        try:
            api.model_info(path)
        except Exception:
            return False
    return True


class TestDSparkTPLosslessQwen3(CustomTestCase, GSM8KMixin):
    """DSpark tp_size=2 cross-rank determinism + GSM8K accuracy floor. GPU-only.

    Launches a single tp=2 DSpark spec server (cuda graph disabled, no overlap)
    and asserts the two sound TP losslessness invariants: identical text for
    repeated greedy requests, and a GSM8K accuracy / accept-length floor.
    """

    target_model = DEFAULT_TARGET_MODEL_DSPARK_QWEN3
    draft_model = DEFAULT_DRAFT_MODEL_DSPARK_QWEN3
    attention_backend = "flashinfer"
    mem_fraction_static = 0.7

    # Lossless greedy keeps tp=2 accuracy at the target model's own level
    # (measured ~0.88 for Qwen3-4B); 0.75 is a comfortable floor. Accept length
    # must clear 1.0 to prove speculation accepts drafts (measured ~4.1).
    gsm8k_score_threshold = 0.75
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0

    @property
    def model(self) -> str:
        return self.target_model

    @classmethod
    def _spec_args(cls, tp_size: int) -> list:
        return [
            "--trust-remote-code",
            "--attention-backend",
            cls.attention_backend,
            "--mem-fraction-static",
            str(cls.mem_fraction_static),
            "--page-size",
            "1",
            "--disable-cuda-graph",
            "--tp",
            str(tp_size),
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--disable-overlap-schedule",
        ]

    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoints_available = True
        if not cls.target_model or not cls.draft_model:
            return
        if not _checkpoints_available(cls.target_model, cls.draft_model):
            cls.checkpoints_available = False
            return
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.target_model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._spec_args(tp_size=2),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _maybe_skip(self) -> None:
        if not self.target_model or not self.draft_model:
            self.skipTest(
                "Model paths not configured. Set target_model and draft_model."
            )
        if not getattr(self, "checkpoints_available", True):
            self.skipTest(
                f"Checkpoint(s) unavailable (gated/missing/offline): "
                f"{self.target_model}, {self.draft_model}."
            )
        if not hasattr(self, "process"):
            self.skipTest("Server not launched (setUpClass failed or skipped).")

    def test_tp2_greedy_determinism(self):
        """DSpark tp=2 greedy output must be deterministic (cross-rank consistent)."""
        self._maybe_skip()
        prompt = "The capital of Germany is"
        out1 = _greedy_request(self.base_url, prompt)
        out2 = _greedy_request(self.base_url, prompt)
        self.assertEqual(out1, out2, "DSpark tp=2 greedy output is not deterministic.")
        self.assertIsNone(self.process.poll())

    def test_gsm8k(self):
        """DSpark tp=2 GSM8K accuracy and accept length clear the lossless floor."""
        self._maybe_skip()
        super().test_gsm8k()


if __name__ == "__main__":
    unittest.main()
