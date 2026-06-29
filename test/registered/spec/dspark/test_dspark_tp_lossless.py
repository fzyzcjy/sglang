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

register_cuda_ci(est_time=900, stage="base-b", runner_config="2-gpu-large")

DEFAULT_TARGET_MODEL_DSPARK_QWEN3 = "Qwen/Qwen3-8B"
DEFAULT_DRAFT_MODEL_DSPARK_QWEN3 = "deepseek-ai/dspark_qwen3_8b_block7"

_PARITY_PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
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


class TestDSparkTPLosslessQwen3(CustomTestCase):
    """DSpark tp_size=2 greedy output must equal the tp_size=1 baseline.

    Per tp-plan-v3 §8: a TP run must stay cross-rank consistent and remain
    token-for-token identical to the single-GPU baseline under greedy decoding
    (base-logits all-gather makes every rank hold the full vocab). The base
    launches the tp=1 spec server first, captures greedy outputs, tears it down,
    then launches the tp=2 spec server and compares. GPU-only (2 GPUs).
    """

    target_model = DEFAULT_TARGET_MODEL_DSPARK_QWEN3
    draft_model = DEFAULT_DRAFT_MODEL_DSPARK_QWEN3
    attention_backend = "flashinfer"
    mem_fraction_static = 0.7

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
        base_url = DEFAULT_URL_FOR_TEST

        tp1_proc = popen_launch_server(
            cls.target_model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._spec_args(tp_size=1),
        )
        try:
            cls.tp1_outputs = {p: _greedy_request(base_url, p) for p in _PARITY_PROMPTS}
        finally:
            kill_process_tree(tp1_proc.pid)

        cls.base_url = base_url
        cls.process = popen_launch_server(
            cls.target_model,
            base_url,
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

    def test_tp2_greedy_equals_tp1_baseline(self):
        """DSpark tp=2 greedy output must equal the tp=1 baseline token-for-token."""
        self._maybe_skip()
        for prompt in _PARITY_PROMPTS:
            tp2_out = _greedy_request(self.base_url, prompt)
            self.assertEqual(
                tp2_out,
                self.tp1_outputs[prompt],
                f"DSpark tp=2 != tp=1 baseline for prompt {prompt!r}",
            )

    def test_tp2_greedy_determinism(self):
        """DSpark tp=2 greedy output must be deterministic (cross-rank consistent)."""
        self._maybe_skip()
        prompt = "The capital of Germany is"
        out1 = _greedy_request(self.base_url, prompt)
        out2 = _greedy_request(self.base_url, prompt)
        self.assertEqual(out1, out2, "DSpark tp=2 greedy output is not deterministic.")
        self.assertIsNone(self.process.poll())


if __name__ == "__main__":
    unittest.main()
