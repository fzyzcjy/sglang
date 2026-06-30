"""DSpark losslessness: token-for-token parity (eager exact + cuda-graph near-tie).

DSpark greedy decoding is lossless by construction: ``_accept_greedy`` accepts a
draft token only when it equals the target model's argmax and the bonus token is
the target's argmax at the divergence point, so the emitted greedy sequence is
exactly the target model's greedy (argmax) sequence (the canonical DFlash verify
rule, same family as EAGLE). This module proves that empirically with two strict
token-for-token checks against a non-spec reference, in two regimes:

  * EAGER, SHORT (32 tokens): exact equality. The spec verify forward runs the
    target on a gamma+1 token window through a verify-mask attention kernel,
    while non-spec decode runs one token through the decode kernel. Those are
    different kernels, so their logits differ by accumulated fp rounding. At 32
    eager tokens the curated prompts stay below the first near-tie, so the
    reference is exactly reproducible and DSpark matches it token-for-token
    (measured 4/4). This is the genuine deterministic-regime losslessness proof.

  * CUDA GRAPH, LONGER (48 tokens): near-tie-tolerant equality. This exercises
    the default production path. Exact equality cannot be required here -- once a
    greedy run is long enough it accumulates a verify-vs-decode near-tie (top-2
    logit gap <~0.25 nats) that flips the argmax; the non-spec reference is
    itself not bit-reproducible past that point. So instead we require that every
    per-token divergence from the cuda-graph reference falls on a reference
    near-tie (top-2 gap <= ``_NEAR_TIE_EPS``) and FAIL on any divergence at a
    confident reference token (gap > eps), which a lossless accept path cannot
    produce. Measured: divergences are all <= 0.125 nats, none confident.

The resident server is the default cuda-graph spec server (production path); the
eager exact-match outputs are captured from short-lived eager servers in
setUpClass. Aggregate quality is additionally floored the EAGLE/DFlash way: a
GSM8K accuracy threshold set just below the measured non-spec baseline plus a
speculative accept-length floor well above 1 (``GSM8KMixin``).
"""

import unittest

import openai
import requests

from sglang.srt.utils import kill_process_tree
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

register_cuda_ci(est_time=1200, stage="base-b", runner_config="1-gpu-large")

DEFAULT_TARGET_MODEL_DSPARK_QWEN3 = "Qwen/Qwen3-4B"
DEFAULT_DRAFT_MODEL_DSPARK_QWEN3 = "deepseek-ai/dspark_qwen3_4b_block7"

DEFAULT_TARGET_MODEL_DSPARK_GEMMA4 = "google/gemma-4-12B-it"
DEFAULT_DRAFT_MODEL_DSPARK_GEMMA4 = "deepseek-ai/dspark_gemma4_12b_block7"

# Eager exact-match length: at 32 eager tokens the curated prompts stay below the
# first verify-vs-decode near-tie, where the non-spec reference is exactly
# bit-reproducible (measured eager DSpark vs eager non-spec: 4/4 exact at 32).
_EAGER_EXACT_MAX_NEW_TOKENS = 32

# Cuda-graph near-tie length: long enough to enter the regime where exact
# equality is unachievable (the reference itself is not bit-reproducible), so the
# near-tie-tolerant check does the work (measured cuda-graph: 11/12 exact + 1
# near-tie of 0.125 nats at 48; tp2: 9/12 exact + 3 near-ties of 0.125 nats).
_NEAR_TIE_MAX_NEW_TOKENS = 48

# Confident-divergence threshold (nats). All measured divergences are <= 0.125
# nats; 0.3 leaves margin while staying well below a confident token. A divergence
# with a reference top-2 gap above this is a real losslessness bug, not fp noise.
_NEAR_TIE_EPS = 0.3

# Curated greedy prompts whose 32-token eager completion is bit-reproducible.
_EXACT_PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "def fibonacci(n):",
    "The three primary colors are red, green, and",
]

# Broader set for the cuda-graph near-tie check (no exact-equality requirement).
_NEAR_TIE_PROMPTS = _EXACT_PROMPTS + [
    "The future of artificial intelligence is",
    "In a world where robots coexist with humans,",
    "The president of the United States is",
    "Hello, my name is",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
    "To make a peanut butter sandwich, first you",
    "The theory of relativity was developed by",
]

_TEMPERATURE_PROMPT = "The future of artificial intelligence is"


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


class _DSparkLosslessBase(CustomTestCase, GSM8KMixin):
    """Base: sequential eager + cuda-graph reference/spec servers, then resident spec.

    Subclasses set ``target_model``, ``draft_model``, and ``disable_overlap``.
    setUpClass runs four sequential phases (one model resident at a time):

      1. eager non-spec reference -> capture exact greedy@32 -> teardown
      2. eager DSpark spec        -> capture exact greedy@32 -> teardown
      3. cuda-graph non-spec ref  -> capture per-token pieces/gaps@48 -> teardown
      4. cuda-graph DSpark spec   -> stays resident for the live tests

    The eager exact check compares the phase-1 vs phase-2 captures; the cuda-graph
    near-tie check compares the resident spec server against the phase-3 capture.
    The spec server's accuracy is floored via ``GSM8KMixin``.
    """

    # Abstract base: not collected as a test on its own (subclasses set models).
    __test__ = False

    target_model: str = ""
    draft_model: str = ""
    other_launch_args: list = []
    attention_backend: str = "flashinfer"
    disable_overlap: bool = False
    mem_fraction_static: float = 0.7
    # Extra env for the SPEC launches only (e.g. SGLANG_RAGGED_VERIFY_MODE=compact).
    spec_env = None

    # GSM8KMixin knobs. Lossless greedy keeps spec accuracy at the non-spec
    # baseline's level (measured baseline ~0.875, spec ~0.865-0.88 at 200q). 0.84
    # is a tight floor set just below the measured baseline -- far above the old
    # 0.75 and high enough that a broken accept path (which drops accuracy) fails.
    # Accept length must clear 3.0 to prove speculation is genuinely accepting
    # drafts (measured GSM8K-dominated ~4.1, vs a no-op floor of 1.0).
    gsm8k_score_threshold = 0.84
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 3.0

    @property
    def model(self) -> str:
        return self.target_model

    @classmethod
    def _base_args(cls, *, cuda_graph: bool) -> list:
        args = [
            "--trust-remote-code",
            "--attention-backend",
            cls.attention_backend,
            "--mem-fraction-static",
            str(cls.mem_fraction_static),
            "--page-size",
            "1",
        ]
        if not cuda_graph:
            args.append("--disable-cuda-graph")
        return args

    @classmethod
    def _spec_args(cls, *, cuda_graph: bool) -> list:
        args = cls._base_args(cuda_graph=cuda_graph) + [
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
    def _launch(cls, other_args: list, env=None):
        return popen_launch_server(
            cls.target_model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def setUpClass(cls) -> None:
        # No local-snapshot pre-check: launch directly and let HF download on the
        # spot. A gated/offline/unreachable model fails the launch loudly rather
        # than silently skipping (which would give a false green for a lossless run).
        if not cls.target_model or not cls.draft_model:
            return
        url = DEFAULT_URL_FOR_TEST
        cls.base_url = url

        eager_ref = cls._launch(cls._base_args(cuda_graph=False))
        try:
            cls.eager_ref_exact = {
                p: greedy_request(url, p, _EAGER_EXACT_MAX_NEW_TOKENS)
                for p in _EXACT_PROMPTS
            }
        finally:
            kill_process_tree(eager_ref.pid)

        eager_spec = cls._launch(cls._spec_args(cuda_graph=False), env=cls.spec_env)
        try:
            cls.eager_spec_exact = {
                p: greedy_request(url, p, _EAGER_EXACT_MAX_NEW_TOKENS)
                for p in _EXACT_PROMPTS
            }
        finally:
            kill_process_tree(eager_spec.pid)

        cg_ref = cls._launch(cls._base_args(cuda_graph=True))
        try:
            cls.cg_ref_capture = {
                p: capture_reference(url, p, _NEAR_TIE_MAX_NEW_TOKENS)
                for p in _NEAR_TIE_PROMPTS
            }
        finally:
            kill_process_tree(cg_ref.pid)

        cls.process = cls._launch(cls._spec_args(cuda_graph=True), env=cls.spec_env)

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

    def test_eager_greedy_parity_exact(self):
        """Eager DSpark greedy output equals the eager non-spec baseline exactly @32."""
        self._maybe_skip()
        for prompt in _EXACT_PROMPTS:
            self.assertEqual(
                self.eager_spec_exact[prompt],
                self.eager_ref_exact[prompt],
                f"Eager DSpark spec != eager ref for prompt {prompt!r}",
            )

    def test_cuda_graph_greedy_parity_near_tie(self):
        """Every cuda-graph DSpark divergence from the reference is a reference near-tie.

        Exact equality is unachievable past the first verify-vs-decode near-tie,
        so the strict invariant is: any per-token divergence falls on a reference
        top-2 near-tie (gap <= eps); a divergence at a confident reference token
        (gap > eps) is a real losslessness bug and fails.
        """
        self._maybe_skip()
        confident_divergences = []
        near_tie_count = 0
        for prompt in _NEAR_TIE_PROMPTS:
            reference = self.cg_ref_capture[prompt]
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
            f"DSpark diverged from the non-spec reference at a confident token "
            f"(top-2 gap > {_NEAR_TIE_EPS} nats): {confident_divergences!r}. A "
            f"lossless accept path cannot do this.",
        )
        print(
            f"cuda-graph near-tie parity: {near_tie_count} near-tie divergence(s), "
            f"0 confident, over {len(_NEAR_TIE_PROMPTS)} prompts"
        )

    def test_gsm8k(self):
        """DSpark GSM8K accuracy and accept length clear the lossless floor."""
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
        out1 = greedy_request(self.base_url, prompt, _EAGER_EXACT_MAX_NEW_TOKENS)
        out2 = greedy_request(self.base_url, prompt, _EAGER_EXACT_MAX_NEW_TOKENS)
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
        out = _temperature_request(
            self.base_url, _TEMPERATURE_PROMPT, temperature=0.8, top_k=50, top_p=0.95
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


class TestDSparkLosslessQwen3Compact(
    _DSparkLosslessBase,
):
    """DSpark COMPACT (real-N ragged verify) lossless parity for Qwen3.

    Lossless by construction (shares _accept_greedy; the ell_r cutoff only accepts a
    shorter correct prefix), so spec greedy output still equals the non-spec ref.
    Exercises the compact verify path the static subclasses never hit.
    """

    __test__ = True
    # 14B checkpoint (same as test_basic_sanity_dspark); HF downloads on the spot.
    target_model = "Qwen/Qwen3-14B"
    draft_model = "deepseek-ai/dspark_qwen3_14b_block7"
    disable_overlap = False
    spec_env = {"SGLANG_RAGGED_VERIFY_MODE": "compact"}


if __name__ == "__main__":
    unittest.main()
