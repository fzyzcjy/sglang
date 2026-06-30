"""DSpark tp_size=2 losslessness: near-tie token parity + accuracy floor + determinism.

A tp=2 DSpark run must preserve the target model's output under lossless greedy
decoding. The sound strict invariant is token-for-token parity against a tp=2
NON-spec reference (same TP degree -- the comparison is between two tp=2 servers),
tolerant of floating-point near-ties:

  * Tensor parallelism splits each reduction across ranks, so the all-reduce sums
    fp partials in a different order than a single GPU. That reorder flips the
    argmax at any near-tie (top-2 logit gap <~0.25 nats). So a tp=2-vs-tp=1
    comparison is NOT sound for any model (a plain non-spec tp=2 server already
    diverges token-for-token from a non-spec tp=1 server on the first tokens),
    and even tp=2-vs-tp=2 exact equality is not reliable across two separate
    server instances (measured: only 7/12 prompts match exactly at 32 tokens).
  * What IS sound and strict: every per-token divergence of the tp=2 DSpark
    server from the tp=2 non-spec reference must fall on a reference near-tie
    (top-2 gap <= ``_NEAR_TIE_EPS``); a divergence at a confident reference token
    (gap > eps) would be a real losslessness bug. Measured: all divergences are
    <= 0.125 nats, none confident.

Additional sound invariants, mirroring EAGLE/DFlash:
  * cross-rank determinism: two identical greedy requests to the tp=2 server
    return identical text (the ranks stay in lockstep within a run);
  * accuracy floor: GSM8K score (set just below the measured non-spec baseline)
    and speculative accept length clear a threshold well above 1, proving the
    lossless accept path is intact and speculation is actually accepting drafts.
"""

import os
import unittest

from sglang.srt.utils import find_local_repo_dir, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.dspark_lossless_kit import (
    capture_reference,
    first_token_divergence,
    greedy_request,
)
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

# Long enough to enter the regime where exact equality is unachievable; the
# near-tie-tolerant check does the work (measured tp2: 9/12 exact + 3 near-ties
# of 0.125 nats at 48).
_NEAR_TIE_MAX_NEW_TOKENS = 48

# Confident-divergence threshold (nats). All measured tp2 divergences are <= 0.125
# nats; 0.3 leaves margin while staying well below a confident token.
_NEAR_TIE_EPS = 0.3

_NEAR_TIE_PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "def fibonacci(n):",
    "The three primary colors are red, green, and",
    "The future of artificial intelligence is",
    "In a world where robots coexist with humans,",
    "The president of the United States is",
    "Hello, my name is",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
    "To make a peanut butter sandwich, first you",
    "The theory of relativity was developed by",
]


def _checkpoints_available(*model_paths: str) -> bool:
    """True only if every model has a local HF snapshot (cached and launchable).

    A repo existing on the Hub is not enough -- a gated or simply un-cached model
    cannot be launched on the CI runner, which would surface as a setUpClass
    server-launch ERROR rather than a clean skip. Requiring a local snapshot
    skips such models cleanly.
    """
    for path in model_paths:
        if os.path.isdir(path):
            continue
        try:
            snapshot_dir = find_local_repo_dir(path, revision=None)
        except Exception:
            return False
        if not snapshot_dir or not os.path.isdir(snapshot_dir):
            return False
    return True


class TestDSparkTPLosslessQwen3(CustomTestCase, GSM8KMixin):
    """DSpark tp_size=2 near-tie token parity + determinism + GSM8K floor. GPU-only.

    setUpClass launches a tp=2 non-spec reference (eager, cuda graph disabled),
    captures per-token text pieces and top-2 gaps, tears it down, then launches
    the tp=2 DSpark spec server (eager, no overlap) which stays resident for the
    tests: near-tie token parity vs the reference, cross-rank determinism, and a
    GSM8K accuracy / accept-length floor.
    """

    target_model = DEFAULT_TARGET_MODEL_DSPARK_QWEN3
    draft_model = DEFAULT_DRAFT_MODEL_DSPARK_QWEN3
    attention_backend = "flashinfer"
    mem_fraction_static = 0.7

    # Lossless greedy keeps tp=2 accuracy at the non-spec baseline's level
    # (measured tp2 baseline ~0.865, spec ~0.88 at 200q). 0.84 is a tight floor
    # set just below the measured baseline -- far above the old 0.75. Accept
    # length must clear 3.0 to prove speculation accepts drafts (measured
    # GSM8K-dominated ~4.17, vs a no-op floor of 1.0).
    gsm8k_score_threshold = 0.84
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 3.0

    @property
    def model(self) -> str:
        return self.target_model

    @classmethod
    def _base_args(cls, *, spec: bool) -> list:
        args = [
            "--trust-remote-code",
            "--attention-backend",
            cls.attention_backend,
            "--mem-fraction-static",
            str(cls.mem_fraction_static),
            "--page-size",
            "1",
            "--disable-cuda-graph",
            "--tp",
            "2",
        ]
        if spec:
            args += [
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                cls.draft_model,
                "--disable-overlap-schedule",
            ]
        return args

    @classmethod
    def _launch(cls, other_args: list):
        return popen_launch_server(
            cls.target_model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoints_available = True
        if not cls.target_model or not cls.draft_model:
            return
        if not _checkpoints_available(cls.target_model, cls.draft_model):
            cls.checkpoints_available = False
            return
        url = DEFAULT_URL_FOR_TEST
        cls.base_url = url

        ref = cls._launch(cls._base_args(spec=False))
        try:
            cls.ref_capture = {
                p: capture_reference(url, p, _NEAR_TIE_MAX_NEW_TOKENS)
                for p in _NEAR_TIE_PROMPTS
            }
        finally:
            kill_process_tree(ref.pid)

        cls.process = cls._launch(cls._base_args(spec=True))

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

    def test_tp2_greedy_parity_near_tie(self):
        """Every tp=2 DSpark divergence from the tp=2 non-spec reference is a near-tie.

        tp=2-vs-tp=2 exact equality is not reliable (all-reduce fp reorder flips
        near-tie argmaxes across server instances), so the strict invariant is:
        any per-token divergence falls on a reference top-2 near-tie (gap <= eps);
        a divergence at a confident reference token (gap > eps) is a real
        losslessness bug and fails.
        """
        self._maybe_skip()
        confident_divergences = []
        near_tie_count = 0
        for prompt in _NEAR_TIE_PROMPTS:
            reference = self.ref_capture[prompt]
            spec_out = greedy_request(self.base_url, prompt, _NEAR_TIE_MAX_NEW_TOKENS)
            divergence = first_token_divergence(reference, spec_out)
            if divergence is None:
                continue
            if divergence.top2_gap > _NEAR_TIE_EPS:
                confident_divergences.append((prompt, divergence))
            else:
                near_tie_count += 1
        self.assertEqual(
            confident_divergences,
            [],
            f"DSpark tp=2 diverged from the tp=2 non-spec reference at a confident "
            f"token (top-2 gap > {_NEAR_TIE_EPS} nats): {confident_divergences!r}. "
            f"A lossless accept path cannot do this.",
        )
        print(
            f"tp=2 near-tie parity: {near_tie_count} near-tie divergence(s), "
            f"0 confident, over {len(_NEAR_TIE_PROMPTS)} prompts"
        )

    def test_tp2_greedy_determinism(self):
        """DSpark tp=2 greedy output must be deterministic (cross-rank consistent)."""
        self._maybe_skip()
        prompt = "The capital of Germany is"
        out1 = greedy_request(self.base_url, prompt, 32)
        out2 = greedy_request(self.base_url, prompt, 32)
        self.assertEqual(out1, out2, "DSpark tp=2 greedy output is not deterministic.")
        self.assertIsNone(self.process.poll())

    def test_gsm8k(self):
        """DSpark tp=2 GSM8K accuracy and accept length clear the lossless floor."""
        self._maybe_skip()
        super().test_gsm8k()


if __name__ == "__main__":
    unittest.main()
