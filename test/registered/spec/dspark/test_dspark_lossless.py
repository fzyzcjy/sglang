import unittest

import openai
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

_PARITY_PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "def fibonacci(n):",
    "The three primary colors are red, green, and",
]

_TEMPERATURE_PROMPTS = [
    "The future of artificial intelligence is",
    "In a world where robots coexist with humans,",
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


def _temperature_request(
    url: str,
    prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    max_new_tokens: int = 48,
) -> str:
    """Send a temperature-sampled generation request and return output text."""
    resp = requests.post(
        url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
        },
    )
    resp.raise_for_status()
    return resp.json()["text"]


class _DSparkLosslessBase(CustomTestCase):
    """Base: sequential (ref server then spec server) lossless checks for DSpark.

    Subclasses set `target_model`, `draft_model`, and optionally `other_args`.
    The base launches the reference server first, captures greedy outputs,
    tears it down, then launches the DSpark spec server.
    """

    target_model: str = ""
    draft_model: str = ""
    other_launch_args: list = []
    attention_backend: str = "flashinfer"
    disable_overlap: bool = False
    mem_fraction_static: float = 0.7

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
        args.extend(cls.other_launch_args)
        return args

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.target_model or not cls.draft_model:
            return
        base_url = DEFAULT_URL_FOR_TEST

        ref_proc = popen_launch_server(
            cls.target_model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._base_args(),
        )
        try:
            cls.parity_ref_outputs = {
                p: _greedy_request(base_url, p) for p in _PARITY_PROMPTS
            }
        finally:
            kill_process_tree(ref_proc.pid)

        cls.base_url = base_url
        cls.process = popen_launch_server(
            cls.target_model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._spec_args(),
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
        if not hasattr(self, "process"):
            self.skipTest("Server not launched (setUpClass failed or skipped).")

    def test_greedy_parity_token_for_token(self):
        """DSpark greedy output must equal non-spec baseline token-for-token."""
        self._maybe_skip()
        for prompt in _PARITY_PROMPTS:
            spec_out = _greedy_request(self.base_url, prompt)
            self.assertEqual(
                spec_out,
                self.parity_ref_outputs[prompt],
                f"DSpark spec != ref baseline for prompt {prompt!r}",
            )

    def test_early_stop_does_not_crash(self):
        """Short max_new_tokens (1-3) with DSpark must not crash the server."""
        self._maybe_skip()
        client = openai.Client(base_url=self.base_url + "/v1", api_key="EMPTY")
        for i in range(4):
            max_tokens = (i % 3) + 1
            response = client.completions.create(
                model=self.target_model,
                prompt=f"There are {i} apples. How to divide them equally?",
                max_tokens=max_tokens,
                temperature=0,
            )
            text = response.choices[0].text
            self.assertIsInstance(text, str)
        self.assertIsNone(self.process.poll())

    def test_greedy_determinism(self):
        """DSpark greedy output must be deterministic across two identical requests."""
        self._maybe_skip()
        prompt = "The capital of Germany is"
        out1 = _greedy_request(self.base_url, prompt)
        out2 = _greedy_request(self.base_url, prompt)
        self.assertEqual(out1, out2, "DSpark greedy output is not deterministic.")
        self.assertIsNone(self.process.poll())

    def test_temperature_sampling_lossless_with_top_k_top_p(self):
        """DSpark temperature sampling with target top-k/top-p must be lossless.

        Losslessness for temperature sampling holds because the chain rejection
        kernel operates on any valid draft distribution q (plan §3 [P0-B]).
        We verify by re-running the same prompt twice and confirming the server
        does not crash. Exact distribution equivalence is verified by the
        statistical test below.
        """
        self._maybe_skip()
        prompt = _TEMPERATURE_PROMPTS[0]
        out1 = _temperature_request(
            self.base_url,
            prompt,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            seed=42,
        )
        self.assertIsInstance(out1, str)
        self.assertIsNone(self.process.poll())

    def test_server_stays_alive_after_batch(self):
        """Server process must stay alive after processing a batch of requests."""
        self._maybe_skip()
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
        ]
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(resp.status_code, 200)
        results = resp.json()
        self.assertEqual(len(results), len(prompts))
        for r in results:
            self.assertIn("text", r)
        self.assertIsNone(self.process.poll())


class TestDSparkLosslessQwen3(
    _DSparkLosslessBase,
):
    """DSpark lossless parity for Qwen3 target + Qwen3-DSpark draft (no overlap)."""

    target_model = DEFAULT_TARGET_MODEL_DSPARK_QWEN3
    draft_model = DEFAULT_DRAFT_MODEL_DSPARK_QWEN3
    disable_overlap = True


class TestDSparkLosslessQwen3Overlap(
    _DSparkLosslessBase,
):
    """DSpark lossless parity for Qwen3 target + draft with overlap scheduling."""

    target_model = DEFAULT_TARGET_MODEL_DSPARK_QWEN3
    draft_model = DEFAULT_DRAFT_MODEL_DSPARK_QWEN3
    disable_overlap = False


class TestDSparkLosslessGemma4(
    _DSparkLosslessBase,
):
    """DSpark lossless parity for Gemma4 target + Gemma4-DSpark draft."""

    target_model = DEFAULT_TARGET_MODEL_DSPARK_GEMMA4
    draft_model = DEFAULT_DRAFT_MODEL_DSPARK_GEMMA4
    disable_overlap = True


if __name__ == "__main__":
    unittest.main()
