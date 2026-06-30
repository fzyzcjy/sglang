import functools
import os
import sys
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

_CUDA_AVAILABLE = torch.cuda.is_available()

# Granularity-A components are pure linear (no through-sparse-attention fp8), so the
# dense parity tolerance (test_dspark_model_parity.py) applies tight.
_ATOL = 1e-4
_RTOL = 1e-4

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


def _ensure_repo_test_package() -> None:
    """Put repo root on sys.path and evict the stdlib ``test`` package shadow."""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    for name in [m for m in list(sys.modules) if m == "test" or m.startswith("test.")]:
        module = sys.modules.get(name)
        file = getattr(module, "__file__", "") or ""
        if not file.startswith(_REPO_ROOT + os.sep):
            del sys.modules[name]


def _requires_cuda(test_method):
    """Decorator: skip when CUDA is unavailable (the dsv4 backbone is GPU-only)."""

    @functools.wraps(test_method)
    def wrapper(self, *args, **kwargs):
        if not _CUDA_AVAILABLE:
            self.skipTest("CUDA not available; dsv4 sparse backend is GPU-only.")
        return test_method(self, *args, **kwargs)

    return wrapper


def _setup_sglang_runtime() -> None:
    """Set global server args and a single-rank model-parallel group (tp=1).

    The dsv4 DSpark backbone builds VocabParallelEmbedding / TP-sharded projections
    that require the model-parallel group, and reads get_global_server_args; both must
    exist before the SGLang model is constructed. Mirrors the dense parity test.
    """
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29653")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl")
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="nccl",
        )


class TestDsv4ComponentParity(CustomTestCase):
    """Granularity-A (category 1): per-component dsv4 draft parity vs the SoT oracle.

    Builds the SGLang ``DeepseekV4ForCausalLMDSpark`` and the vendored
    ``RefTransformer`` with identical random weights (``sync_sot_to_sgl_dsv4``), then
    feeds each pure-linear component the SAME input and asserts the outputs match
    tight (1e-4, greedy exact). These avoid the through-sparse-attention fp8 path
    (granularity B), so they isolate the MLA-LoRA weight mapping and the head math:

      * kv-projection (isolates the MLA-LoRA weight mapping FIRST);
      * target-hidden projection (main_proj + main_norm);
      * hc_head collapse (the dsv4-specific PRE-norm collapse the dense path lacks);
      * markov head (serial bias-then-sample on the collapsed base logits);
      * confidence head (post-hc_head PRE-norm tap).
    """

    enable_confidence_head: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        cls._skip_reason = None
        if not _CUDA_AVAILABLE:
            cls._skip_reason = "CUDA not available; dsv4 backbone is GPU-only."
            return
        try:
            _ensure_repo_test_package()
            from test.srt.speculative._dspark_reference.deepseek_v4.modeling import (
                RefTransformer,
            )
            from test.srt.speculative._dspark_reference.deepseek_v4.parity_fixture import (
                attach_shared_modules_from_ref,
                force_native_ops,
                make_ref_args_from_config,
                make_tiny_dsv4_config,
                sync_sot_to_sgl_dsv4,
            )

            from sglang.srt.models.deepseek_v4_dspark import (
                DeepseekV4ForCausalLMDSpark,
            )
            from sglang.srt.runtime_context import get_parallel
        except ImportError as exc:
            cls._skip_reason = f"Import error: {exc}"
            return

        _setup_sglang_runtime()
        device = torch.device("cuda")
        cls.device = device
        config = make_tiny_dsv4_config(
            enable_confidence_head=cls.enable_confidence_head
        )
        cls.config = config

        torch.manual_seed(60)
        ref_args = make_ref_args_from_config(config)
        ref = RefTransformer(ref_args).to(device).eval()
        with get_parallel().override(tp_size=1, tp_rank=0):
            sgl = DeepseekV4ForCausalLMDSpark(config=config).to(device).eval()
        force_native_ops(sgl)

        sync_sot_to_sgl_dsv4(ref=ref, sgl=sgl, config=config)
        attach_shared_modules_from_ref(sgl=sgl, ref=ref, config=config, device=device)

        cls.ref = ref
        cls.sgl = sgl

    def _skip_if_unready(self) -> None:
        if self._skip_reason:
            self.skipTest(self._skip_reason)

    @_requires_cuda
    def test_kv_projection_parity(self) -> None:
        """SGLang kv_proj_only + kv_norm matches the SoT kv_norm(wkv(.)) (weight map)."""
        self._skip_if_unready()
        config = self.config
        ctx_len = 5
        torch.manual_seed(61)
        ctx_hidden = torch.randn(ctx_len, config.hidden_size, device=self.device)

        for stage_id, sgl_stage in enumerate(self.sgl.stages):
            attn = sgl_stage.self_attn
            ref_attn = self.ref.mtp[stage_id].attn
            with torch.no_grad():
                sgl_kv = attn.kv_norm(attn.kv_proj_only(ctx_hidden))
                ref_kv = ref_attn.kv_norm(ref_attn.wkv(ctx_hidden))
            torch.testing.assert_close(
                sgl_kv.float(), ref_kv.float(), atol=_ATOL, rtol=_RTOL
            )

    @_requires_cuda
    def test_target_hidden_projection_parity(self) -> None:
        """SGLang project_target_hidden matches the SoT main_norm(main_proj(.))."""
        self._skip_if_unready()
        config = self.config
        ctx_len = 5
        n_features = len(config.dspark_target_layer_ids)
        torch.manual_seed(62)
        main_hidden = torch.randn(
            ctx_len, n_features * config.hidden_size, device=self.device
        )

        with torch.no_grad():
            sgl_ctx = self.sgl.project_target_hidden(main_hidden)
            ref_ctx = self.ref.project_target_hidden(main_hidden)
        self.assertEqual(sgl_ctx.shape, (ctx_len, config.hidden_size))
        torch.testing.assert_close(
            sgl_ctx.float(), ref_ctx.float(), atol=_ATOL, rtol=_RTOL
        )

    @_requires_cuda
    def test_hc_head_collapse_parity(self) -> None:
        """SGLang collapse_hc_head matches the SoT hc_head(.) (the PRE-norm collapse)."""
        self._skip_if_unready()
        config = self.config
        n = 6
        torch.manual_seed(63)
        x = torch.randn(n, config.hc_mult, config.hidden_size, device=self.device)

        with torch.no_grad():
            sgl_post_hc = self.sgl.collapse_hc_head(x)
            ref_post_hc = self.ref.collapse_hc_head(x.unsqueeze(0)).squeeze(0)
        self.assertEqual(sgl_post_hc.shape, (n, config.hidden_size))
        torch.testing.assert_close(
            sgl_post_hc.float(), ref_post_hc.float(), atol=_ATOL, rtol=_RTOL
        )

    @_requires_cuda
    def test_base_logits_from_hidden_parity(self) -> None:
        """SGLang hc_head -> norm -> lm_head matches the SoT full-vocab base logits."""
        self._skip_if_unready()
        config = self.config
        n = 6
        torch.manual_seed(64)
        x = torch.randn(n, config.hc_mult, config.hidden_size, device=self.device)

        with torch.no_grad():
            sgl_logits = self.sgl.compute_base_logits(x)
            ref_logits = self.ref.base_logits_from_hidden(x.unsqueeze(0)).squeeze(0)
        self.assertEqual(sgl_logits.shape, (n, config.vocab_size))
        torch.testing.assert_close(
            sgl_logits.float(), ref_logits.float(), atol=_ATOL, rtol=_RTOL
        )
        torch.testing.assert_close(sgl_logits.argmax(dim=-1), ref_logits.argmax(dim=-1))

    @_requires_cuda
    def test_markov_head_serial_correction_parity(self) -> None:
        """SGLang Markov sample_block matches the SoT serial bias-then-sample loop."""
        self._skip_if_unready()
        config = self.config
        bs = 2
        gamma = config.dspark_block_size
        vocab = config.vocab_size
        torch.manual_seed(65)
        base_logits = torch.randn(bs, gamma, vocab, device=self.device)
        first_prev = torch.randint(0, vocab, (bs,), device=self.device)

        def greedy_sampler(step_logits, step_idx):
            return step_logits.argmax(dim=-1)

        with torch.no_grad():
            sgl_sampled, sgl_corrected = self.sgl.markov_head.sample_block(
                base_logits,
                first_prev_tokens=first_prev,
                hidden_states=None,
                sampler=greedy_sampler,
            )
            ref_head = self.ref.mtp[-1].markov_head
            ref_corrected = []
            prev = first_prev.long()
            for step_idx in range(gamma):
                bias, _ = ref_head(prev)
                step_logits = base_logits[:, step_idx, :] + bias
                ref_corrected.append(step_logits.unsqueeze(1))
                prev = step_logits.argmax(dim=-1)
            ref_corrected = torch.cat(ref_corrected, dim=1)

        torch.testing.assert_close(
            sgl_corrected.float(), ref_corrected.float(), atol=_ATOL, rtol=_RTOL
        )
        torch.testing.assert_close(sgl_sampled, ref_corrected.argmax(dim=-1))


class TestDsv4ComponentParityWithConfidence(TestDsv4ComponentParity):
    """Granularity-A parity with the confidence head enabled (post-hc_head PRE-norm tap)."""

    enable_confidence_head = True

    @_requires_cuda
    def test_confidence_head_parity(self) -> None:
        """SGLang compute_confidence matches the SoT confidence on the same tap.

        The dsv4 confidence tap is the post-hc_head PRE-norm draft hidden (reference
        model.py:873: confidence_head(x, markov_embed) where x = hc_head(.) BEFORE the
        norm), with the per-step markov_embed built from the prev-token sequence
        [anchor, s_0, ..., s_{gamma-2}]. The dense worker's post-norm tap would be the
        wrong tap for V4, so this pins the correct one.
        """
        self._skip_if_unready()
        config = self.config
        bs = 2
        gamma = config.dspark_block_size
        vocab = config.vocab_size
        d = config.hidden_size
        torch.manual_seed(66)

        # Stash the post-hc_head PRE-norm tap on the model exactly as a real forward
        # would (compute_base_logits stashes self._x_post_hc), then drive the worker
        # confidence hook with explicit anchor + sampled tokens.
        x_post_hc = torch.randn(bs * gamma, d, device=self.device)
        self.sgl._x_post_hc = x_post_hc
        self.sgl.gamma = gamma
        anchor = torch.randint(0, vocab, (bs,), device=self.device)
        sampled = torch.randint(0, vocab, (bs, gamma), device=self.device)

        with torch.no_grad():
            sgl_conf = self.sgl.compute_confidence(
                anchor_tokens=anchor, sampled_tokens=sampled
            )
            # SoT confidence: cat([x_post_hc, markov_embed], -1) -> proj -> sigmoid.
            ref_head = self.ref.mtp[-1]
            prev_seq = torch.cat([anchor.view(-1, 1), sampled[:, : gamma - 1]], dim=1)
            ref_markov_embed = ref_head.markov_head.markov_w1(prev_seq)
            ref_x = x_post_hc.view(bs, gamma, d)
            ref_conf_raw = ref_head.confidence_head(ref_x, ref_markov_embed)
            ref_conf = torch.sigmoid(ref_conf_raw.float())

        self.assertIsNotNone(sgl_conf)
        self.assertEqual(sgl_conf.shape, (bs, gamma))
        torch.testing.assert_close(
            sgl_conf.float(), ref_conf.float(), atol=_ATOL, rtol=_RTOL
        )


if __name__ == "__main__":
    unittest.main()
