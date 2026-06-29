"""DSpark losslessness: eager strict token-for-token parity + accuracy floor.

DSpark greedy decoding is lossless by construction: ``_accept_greedy`` accepts a
draft token only when it equals the target model's argmax and the bonus token is
the target's argmax at the divergence point, so the emitted greedy sequence is
exactly the target model's greedy (argmax) sequence (the canonical DFlash verify
rule, same family as EAGLE).

The strong assertion is therefore token-for-token equality against a non-spec
reference -- but only in a regime where the reference is itself reproducible.
That regime is EAGER, SHORT generation:

  * The spec verify forward runs the target on a gamma+1 token window through a
    verify-mask attention kernel, while non-spec decode runs one token through
    the decode kernel. These are different kernels, so their logits differ by
    accumulated fp rounding. At a near-tie (top-2 logit gap <~0.25 nats) that
    rounding flips the argmax, and the divergence then cascades. This is benign
    fp noise -- the non-spec baseline is ALSO not bit-reproducible across runs
    once a long greedy generation accumulates such near-ties (measured: a plain
    non-spec eager server self-diverges by ~128 tokens). It is the same effect
    EAGLE's parity test lives with; EAGLE just stays under the near-tie horizon
    for its prompts.
  * So strict token-for-token is asserted in EAGER mode at a SHORT length
    (32 tokens) where the curated prompts stay below the first near-tie and the
    reference is exactly reproducible. This is the genuine losslessness proof.

Token-for-token equality cannot be extended to many prompts or long generations
(it would hit the near-ties above), so overall output quality is floored the
EAGLE/DFlash way: a GSM8K accuracy threshold plus a speculative accept-length
threshold (``GSM8KMixin``). Lossless greedy keeps spec accuracy equal to the
target model's own accuracy modulo the near-tie noise, so a comfortable absolute
floor is the right invariant -- a broken accept path would drop accuracy or
collapse the accept length.
"""

import os
import unittest

import openai
import requests

from sglang.srt.utils import find_local_repo_dir, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-large")

DEFAULT_TARGET_MODEL_DSPARK_QWEN3 = "Qwen/Qwen3-4B"
DEFAULT_DRAFT_MODEL_DSPARK_QWEN3 = "deepseek-ai/dspark_qwen3_4b_block7"

DEFAULT_TARGET_MODEL_DSPARK_GEMMA4 = "google/gemma-4-12B-it"
DEFAULT_DRAFT_MODEL_DSPARK_GEMMA4 = "deepseek-ai/dspark_gemma4_12b_block7"

# Short generation so the curated greedy prompts stay below the first verify-vs-
# decode near-tie, where the non-spec reference is itself bit-reproducible.
_PARITY_MAX_NEW_TOKENS = 32

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


def _greedy_request(url: str, prompt: str, max_new_tokens: int) -> str:
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
    max_new_tokens: int = 48,
) -> str:
    """Send a temperature-sampled generation request and return the output text."""
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


class _DSparkLosslessBase(CustomTestCase, GSM8KMixin):
    """Base: sequential eager reference then eager DSpark spec server.

    Subclasses set ``target_model``, ``draft_model``, and optionally
    ``disable_overlap``. The base launches the eager reference server first,
    captures short greedy outputs, tears it down, then launches the eager DSpark
    spec server (one model resident at a time). Both servers run with cuda graph
    disabled so the token-for-token parity check is in a reproducible regime; the
    spec server's accuracy is then floored via ``GSM8KMixin``.
    """

    # Abstract base: not collected as a test on its own (subclasses set models).
    __test__ = False

    target_model: str = ""
    draft_model: str = ""
    other_launch_args: list = []
    attention_backend: str = "flashinfer"
    disable_overlap: bool = False
    mem_fraction_static: float = 0.7

    # GSM8KMixin knobs. Lossless greedy keeps spec accuracy at the target model's
    # own level (measured ~0.87 for Qwen3-4B); 0.75 is a comfortable floor that
    # still catches a broken accept path. Accept length must clear 1.0 to prove
    # speculation actually accepts drafts (measured ~4.1).
    gsm8k_score_threshold = 0.75
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0

    @property
    def model(self) -> str:
        return self.target_model

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
            "--disable-cuda-graph",
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
        cls.checkpoints_available = True
        if not cls.target_model or not cls.draft_model:
            return
        if not _checkpoints_available(cls.target_model, cls.draft_model):
            cls.checkpoints_available = False
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
                p: _greedy_request(base_url, p, _PARITY_MAX_NEW_TOKENS)
                for p in _PARITY_PROMPTS
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
        if not getattr(self, "checkpoints_available", True):
            self.skipTest(
                f"Checkpoint(s) unavailable (gated/missing/offline): "
                f"{self.target_model}, {self.draft_model}."
            )
        if not hasattr(self, "process"):
            self.skipTest("Server not launched (setUpClass failed or skipped).")

    def test_greedy_parity_token_for_token(self):
        """Eager DSpark greedy output equals the eager non-spec baseline exactly."""
        self._maybe_skip()
        for prompt in _PARITY_PROMPTS:
            spec_out = _greedy_request(self.base_url, prompt, _PARITY_MAX_NEW_TOKENS)
            self.assertEqual(
                spec_out,
                self.parity_ref_outputs[prompt],
                f"DSpark spec != ref baseline for prompt {prompt!r}",
            )

    def test_gsm8k(self):
        """Default-path GSM8K accuracy and accept length clear the lossless floor."""
        self._maybe_skip()
        super().test_gsm8k()

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
        out1 = _greedy_request(self.base_url, prompt, _PARITY_MAX_NEW_TOKENS)
        out2 = _greedy_request(self.base_url, prompt, _PARITY_MAX_NEW_TOKENS)
        self.assertEqual(out1, out2, "DSpark greedy output is not deterministic.")
        self.assertIsNone(self.process.poll())

    def test_temperature_sampling_runs_and_stays_alive(self):
        """DSpark temperature sampling yields valid text and keeps the server alive.

        Bit-identical repeats are deliberately NOT asserted: the spec rejection
        sampler draws fresh ``torch.rand`` uniforms per verify step and does not
        thread ``sampling_seed`` through that path, so two identical seeded
        temperature requests are not expected to match (the same is true for
        EAGLE/DFlash spec v2). The honest invariant here is that temperature
        decoding produces a well-formed completion and does not crash the server.
        """
        self._maybe_skip()
        prompt = _TEMPERATURE_PROMPTS[0]
        out = _temperature_request(
            self.base_url, prompt, temperature=0.8, top_k=50, top_p=0.95
        )
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)
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

    __test__ = True
    target_model = DEFAULT_TARGET_MODEL_DSPARK_QWEN3
    draft_model = DEFAULT_DRAFT_MODEL_DSPARK_QWEN3
    disable_overlap = True


class TestDSparkLosslessQwen3Overlap(
    _DSparkLosslessBase,
):
    """DSpark lossless parity for Qwen3 target + draft with overlap scheduling."""

    __test__ = True
    target_model = DEFAULT_TARGET_MODEL_DSPARK_QWEN3
    draft_model = DEFAULT_DRAFT_MODEL_DSPARK_QWEN3
    disable_overlap = False


class TestDSparkLosslessGemma4(
    _DSparkLosslessBase,
):
    """DSpark lossless parity for Gemma4 target + Gemma4-DSpark draft."""

    __test__ = True
    target_model = DEFAULT_TARGET_MODEL_DSPARK_GEMMA4
    draft_model = DEFAULT_DRAFT_MODEL_DSPARK_GEMMA4
    disable_overlap = True


if __name__ == "__main__":
    unittest.main()
