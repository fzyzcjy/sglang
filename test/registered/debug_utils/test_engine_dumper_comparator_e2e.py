"""E2E test: source patcher + dumper + comparator on SGLang server.

Patches Qwen3MoeDecoderLayer.forward, fused_experts_impl, DeepEP dispatch,
and GEMM functions to insert dumper.dump() calls, launches servers with Qwen3-30B-A3B
(MOE model), runs inference, verifies patched dump fields exist, then
runs comparator to verify numerical consistency.

Two test classes, each with its own model and shared baseline:

TestBF16 (Qwen3-30B-A3B):
- test_basic_tp: TP=2 baseline vs TP=4 target
- test_dp_attention: TP=2 baseline vs TP=2+DP=2+dp-attention target
- test_ep_fused_moe: TP=2 baseline vs TP=4+EP=4 target (FusedMoE)

TestFP8DeepEP (Qwen3-30B-A3B-FP8):
- test_ep_deepep_normal: TP=2 baseline vs TP=2+DeepEP normal target
- test_ep_deepep_low_latency: TP=2 baseline vs TP=2+DeepEP low-latency target

Known limitations:
- FP8 + TP=4 incompatible: Qwen3-30B-A3B-FP8 has moe_intermediate_size=768,
  TP=4 yields 768/4=192 per rank, which is not divisible by FP8
  weight_block_size[0]=128 -> ValueError in fp8.py:create_weights.
  FP8 tests are therefore limited to TP=2.
- BF16 + DeepEP incompatible: upstream sglang ep_moe/layer.py has
  ``assert False`` in forward_deepgemm_contiguous / forward_deepgemm_masked
  (non-quantized GEMM paths are deprecated). BF16 models hit these asserts;
  FP8 models take the quantized path and bypass them.

The dumper.apply_source_patches() auto-injects ``from ... import dumper``
so the YAML only needs ``dumper.dump(...)`` calls.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pytest
import requests

pytestmark = pytest.mark.filterwarnings(
    "ignore:Unknown config option. asyncio_mode:pytest.PytestConfigWarning",
)

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_cuda_ci(est_time=300, suite="nightly-4-gpu", nightly=True)

MODEL_BF16 = "Qwen/Qwen3-30B-A3B"
MODEL_FP8 = "Qwen/Qwen3-30B-A3B-FP8"
BASELINE_TP = 2
TARGET_TP = 4
EXP_NAME = "e2e_source_patcher"
# Only dump layer 0: its input is the embedding output, which is identical
# across all parallel configurations.  This makes router_logits, topk_ids,
# and gateup_output deterministic (no accumulated numerical drift).
# Layer >= 1 accumulates BF16 drift from the previous layer's TP/EP
# accumulation-order differences, which can flip routing decisions at
# decision boundaries — making comparisons unreliable.
DUMPER_FILTER = "layer_id == 0"

_FIELDS_TO_VERIFY: list[str] = [
    # decoder layer level (aligned with miles)
    "layer_input",
    "attn_output",
    "pre_mlp_residual",
    "mlp_output",
    # attention internals
    "attn_pre_o_proj",
    # moe internals
    "moe_router_logits",
    "moe_topk_ids",
    "moe_expert_output",
]

_FIELDS_GATEUP: list[str] = [
    "gateup_output",
]

PATCH_CONFIG_YAML: str = """\
patches:
  # --- decoder layer level (aligned with miles test) ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeDecoderLayer.forward
    edits:
      - match: |
          hidden_states, residual = (
              self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                  hidden_states,
                  residual,
                  forward_batch,
                  captured_last_layer_outputs=captured_last_layer_outputs,
                  **kwargs,
              )
          )
        append: |
          dumper.dump('layer_input', hidden_states, dims='t h # tp:replicated moe_tp:replicated')
      - match: |
          hidden_states = self.self_attn(
              positions=positions,
              hidden_states=hidden_states,
              forward_batch=forward_batch,
          )
        append: "dumper.dump('attn_output', hidden_states, dims='t h[tp:partial] # moe_tp:replicated')"
      - match: |
          hidden_states, residual = self.layer_communicator.prepare_mlp(
              hidden_states, residual, forward_batch
          )
        append: "dumper.dump('pre_mlp_residual', hidden_states, dims='t h # tp:replicated moe_tp:replicated')"
      - match: |
          hidden_states = self.mlp(
              hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
          )
        append: "dumper.dump('mlp_output', hidden_states, dims='t h[tp:partial] # moe_tp:replicated')"

  # --- attention internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeAttention.forward_core
    edits:
      - match: "output, _ = self.o_proj(attn_output)"
        prepend: "dumper.dump('attn_pre_o_proj', attn_output, dims='t attn_h[tp] # moe_tp:replicated')"

  # --- moe internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward_normal
    edits:
      - match: "router_logits, _ = self.gate(hidden_states)"
        append: "dumper.dump('moe_router_logits', router_logits, dims='t num_experts # tp:replicated moe_tp:replicated')"
      - match: "topk_output = self.topk(hidden_states, router_logits)"
        append: "dumper.dump('moe_topk_ids', topk_output.topk_ids, dims='t top_k # tp:replicated moe_tp:replicated')"
      - match: "final_hidden_states = self.experts(hidden_states, topk_output)"
        append: "dumper.dump('moe_expert_output', final_hidden_states, dims='t h[tp:partial] # moe_tp:replicated')"

  # --- moe expert intermediate (gate/up GEMM output, before activation) ---
  # intermediate_cache1 has shape [T, 2*I_local] where the first I_local cols
  # are gate and last I_local are up (MergedColumnParallel sharding).
  # Simple concat would produce [gate_r0, up_r0, gate_r1, up_r1, ...] which
  # differs between TP=2 and TP=4.  Reshaping to [T, 2, I_local] and
  # annotating only I_local as [moe_tp] yields [T, 2, I_full] after concat,
  # which is order-invariant to moe_tp_size.
  #
  # The [ep] derouter normalises token-dispatch order to canonical order using
  # sorted_token_ids. This is needed when the dispatch order might differ
  # between baseline and target (e.g. dp-attention, where tiny BF16 diffs
  # in router logits can change topk_ids at decision boundaries).
  - target: sglang.srt.layers.moe.fused_moe_triton.fused_moe.fused_experts_impl
    edits:
      - match: |
          sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
              curr_topk_ids, config["BLOCK_SIZE_M"], E
          )
        append: |
          dumper.dump('fused_moe_sorted_token_ids', sorted_token_ids)
          dumper.dump('fused_moe_ep_num_tokens', torch.tensor(tokens_in_chunk))
          dumper.dump('fused_moe_ep_top_k', torch.tensor(topk))
      - match: "# Activation function with multiplication"
        prepend: |
          _ic1_t, _ic1_n = intermediate_cache1.shape
          dumper.dump('gateup_output', intermediate_cache1.view(_ic1_t, 2, _ic1_n // 2), dims='t_k[ep] gate_up h_inter[moe_tp] # tp:replicated')
"""

PATCH_CONFIG_DP_ATTENTION_YAML: str = """\
patches:
  # --- decoder layer level (aligned with miles test) ---
  # In dp-attention mode: attn tensors are NOT TP-sharded (attn_tp_size=1),
  # and mlp_output is already all-reduced inside forward_normal().
  # layer_input is dumped after prepare_attn which DP-distributes tokens,
  # so it needs dp:=attn_dp to filter to the non-empty DP rank.
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeDecoderLayer.forward
    edits:
      - match: |
          hidden_states, residual = (
              self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                  hidden_states,
                  residual,
                  forward_batch,
                  captured_last_layer_outputs=captured_last_layer_outputs,
                  **kwargs,
              )
          )
        append: |
          dumper.dump('layer_input', hidden_states, dims='t h # tp:replicated moe_tp:replicated dp:=attn_dp')
      - match: |
          hidden_states = self.self_attn(
              positions=positions,
              hidden_states=hidden_states,
              forward_batch=forward_batch,
          )
        append: "dumper.dump('attn_output', hidden_states, dims='t h # tp:replicated moe_tp:replicated')"
      - match: |
          hidden_states, residual = self.layer_communicator.prepare_mlp(
              hidden_states, residual, forward_batch
          )
        append: "dumper.dump('pre_mlp_residual', hidden_states, dims='t h # tp:replicated moe_tp:replicated')"
      - match: |
          hidden_states = self.mlp(
              hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
          )
        append: "dumper.dump('mlp_output', hidden_states, dims='t h # tp:replicated moe_tp:replicated')"

  # --- attention internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeAttention.forward_core
    edits:
      - match: "output, _ = self.o_proj(attn_output)"
        prepend: "dumper.dump('attn_pre_o_proj', attn_output, dims='t attn_h # tp:replicated moe_tp:replicated')"

  # --- moe internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward_normal
    edits:
      - match: "router_logits, _ = self.gate(hidden_states)"
        append: "dumper.dump('moe_router_logits', router_logits, dims='t num_experts # tp:replicated moe_tp:replicated')"
      - match: "topk_output = self.topk(hidden_states, router_logits)"
        append: "dumper.dump('moe_topk_ids', topk_output.topk_ids, dims='t top_k # tp:replicated moe_tp:replicated')"
      - match: "final_hidden_states = self.experts(hidden_states, topk_output)"
        append: "dumper.dump('moe_expert_output', final_hidden_states, dims='t h[tp:partial] # moe_tp:replicated')"

  # --- moe expert intermediate (gate/up GEMM output, before activation) ---
  # In dp-attention, the target's attention is DP-distributed, so hidden_states
  # entering MoE can have tiny BF16 differences from baseline.  This may cause
  # different expert dispatch ordering (topk_ids differ at decision boundaries).
  # The [ep] derouter normalises both sides to canonical token order using each
  # side's own sorted_token_ids, making the comparison order-invariant.
  # Both baseline and target have the same moe_tp_size, so the reshape to
  # [T, 2, I_local] with moe_tp concat produces identical column ordering.
  - target: sglang.srt.layers.moe.fused_moe_triton.fused_moe.fused_experts_impl
    edits:
      - match: |
          sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
              curr_topk_ids, config["BLOCK_SIZE_M"], E
          )
        append: |
          dumper.dump('fused_moe_sorted_token_ids', sorted_token_ids)
          dumper.dump('fused_moe_ep_num_tokens', torch.tensor(tokens_in_chunk))
          dumper.dump('fused_moe_ep_top_k', torch.tensor(topk))
      - match: "# Activation function with multiplication"
        prepend: |
          _ic1_t, _ic1_n = intermediate_cache1.shape
          dumper.dump('gateup_output', intermediate_cache1.view(_ic1_t, 2, _ic1_n // 2), dims='t_k[ep] gate_up h_inter[moe_tp] # tp:replicated')
"""

PATCH_CONFIG_EP_YAML: str = """\
patches:
  # --- decoder layer level ---
  # With --ep-size, moe_ep is active. All decoder-level and MOE tensors
  # are replicated across moe_ep (FusedMoE combines results internally).
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeDecoderLayer.forward
    edits:
      - match: |
          hidden_states, residual = (
              self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                  hidden_states,
                  residual,
                  forward_batch,
                  captured_last_layer_outputs=captured_last_layer_outputs,
                  **kwargs,
              )
          )
        append: |
          dumper.dump('layer_input', hidden_states, dims='t h # tp:replicated moe_ep:replicated')
      - match: |
          hidden_states = self.self_attn(
              positions=positions,
              hidden_states=hidden_states,
              forward_batch=forward_batch,
          )
        append: "dumper.dump('attn_output', hidden_states, dims='t h[tp:partial] # moe_ep:replicated')"
      - match: |
          hidden_states, residual = self.layer_communicator.prepare_mlp(
              hidden_states, residual, forward_batch
          )
        append: "dumper.dump('pre_mlp_residual', hidden_states, dims='t h # tp:replicated moe_ep:replicated')"
      - match: |
          hidden_states = self.mlp(
              hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
          )
        append: "dumper.dump('mlp_output', hidden_states, dims='t h[tp:partial] # moe_ep:replicated')"

  # --- attention internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeAttention.forward_core
    edits:
      - match: "output, _ = self.o_proj(attn_output)"
        prepend: "dumper.dump('attn_pre_o_proj', attn_output, dims='t attn_h[tp] # moe_ep:replicated')"

  # --- moe internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward_normal
    edits:
      - match: "router_logits, _ = self.gate(hidden_states)"
        append: "dumper.dump('moe_router_logits', router_logits, dims='t num_experts # tp:replicated moe_ep:replicated')"
      - match: "topk_output = self.topk(hidden_states, router_logits)"
        append: "dumper.dump('moe_topk_ids', topk_output.topk_ids, dims='t top_k # tp:replicated moe_ep:replicated')"
      - match: "final_hidden_states = self.experts(hidden_states, topk_output)"
        append: "dumper.dump('moe_expert_output', final_hidden_states, dims='t h[tp:partial] # moe_ep:replicated')"

  # --- moe expert intermediate (gate/up GEMM output, before activation) ---
  # Same fused_experts_impl patch as baseline, with [moe_ep:partial] (reduce-sum, no derouting).
  - target: sglang.srt.layers.moe.fused_moe_triton.fused_moe.fused_experts_impl
    edits:
      - match: |
          sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
              curr_topk_ids, config["BLOCK_SIZE_M"], E
          )
        append: |
          dumper.dump('fused_moe_sorted_token_ids', sorted_token_ids)
          dumper.dump('fused_moe_ep_num_tokens', torch.tensor(tokens_in_chunk))
          dumper.dump('fused_moe_ep_top_k', torch.tensor(topk))
      - match: "# Activation function with multiplication"
        prepend: |
          _ic1_t, _ic1_n = intermediate_cache1.shape
          dumper.dump('gateup_output', intermediate_cache1.view(_ic1_t, 2, _ic1_n // 2), dims='t_k[moe_ep:partial] gate_up h_inter[moe_tp] # tp:replicated')
"""

# DeepEP uses forward_deepep (not forward_normal), so MoE internals need
# different match targets. The experts call uses keyword args and
# final_hidden_states is the DeepEPMoE output (already EP-combined).
#
# Dims: With DeepEP TP=2+EP=2, moe_ep and tp share the same 2 ranks.
# Unlike FusedMoE EP (where ep is an independent axis), DeepEP's EP
# overlays TP ranks — non-MoE tensors are just tp-sharded as usual,
# and MoE outputs are already EP-combined by DeepEPMoE internally.
# Therefore NO moe_ep annotation is needed (same dims as baseline).
PATCH_CONFIG_DEEPEP_YAML: str = """\
patches:
  # --- decoder layer level ---
  # DeepEP auto-sets ep_size=tp_size. Since EP overlays the same ranks
  # as TP, non-MoE tensors have the same dims as the non-EP baseline.
  # With DeepEP, moe_ep is active (size=tp_size) but tensors are replicated
  # across moe_ep (EP-combined internally). moe_tp_size=tp/ep=1 so not active.
  # DeepEP tests use --dp 2 --enable-dp-attention, so prepare_attn
  # distributes tokens across DP ranks (one rank may be empty).
  # Unlike standard DP-attention where prepare_mlp all-gathers tokens
  # back, DeepEP keeps tokens DP-distributed through the MoE layer
  # (DeepEP's all-to-all handles the distribution internally).
  # Therefore ALL dump points need dp:=attn_dp so the comparator
  # filters to the non-empty DP rank.
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeDecoderLayer.forward
    edits:
      - match: |
          hidden_states, residual = (
              self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                  hidden_states,
                  residual,
                  forward_batch,
                  captured_last_layer_outputs=captured_last_layer_outputs,
                  **kwargs,
              )
          )
        append: |
          dumper.dump('layer_input', hidden_states, dims='t h # tp:replicated moe_ep:replicated dp:=attn_dp')
      - match: |
          hidden_states = self.self_attn(
              positions=positions,
              hidden_states=hidden_states,
              forward_batch=forward_batch,
          )
        append: "dumper.dump('attn_output', hidden_states, dims='t h[tp:partial] # moe_ep:replicated dp:=attn_dp')"
      - match: |
          hidden_states, residual = self.layer_communicator.prepare_mlp(
              hidden_states, residual, forward_batch
          )
        append: "dumper.dump('pre_mlp_residual', hidden_states, dims='t h # tp:replicated moe_ep:replicated dp:=attn_dp')"
      - match: |
          hidden_states = self.mlp(
              hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
          )
        append: "dumper.dump('mlp_output', hidden_states, dims='t h[tp:partial] # moe_ep:replicated dp:=attn_dp')"

  # --- attention internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeAttention.forward_core
    edits:
      - match: "output, _ = self.o_proj(attn_output)"
        prepend: "dumper.dump('attn_pre_o_proj', attn_output, dims='t attn_h[tp] # moe_ep:replicated dp:=attn_dp')"

  # --- moe internals (forward_deepep path) ---
  # DeepEPMoE combines EP results internally, so moe_expert_output is
  # already the full (EP-combined) result. Same dims as non-EP baseline.
  # Tokens are still DP-distributed in forward_deepep, so dp:=attn_dp
  # is needed to filter to the non-empty DP rank.
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward_deepep
    edits:
      - match: "router_logits, _ = self.gate(hidden_states)"
        append: "dumper.dump('moe_router_logits', router_logits, dims='t num_experts # tp:replicated moe_ep:replicated dp:=attn_dp')"
      - match: |
          topk_output = self.topk(
              hidden_states,
              router_logits,
              num_token_non_padded=forward_batch.num_token_non_padded,
              expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                  layer_id=self.layer_id,
              ),
          )
        append: "dumper.dump('moe_topk_ids', topk_output.topk_ids, dims='t top_k # tp:replicated moe_ep:replicated dp:=attn_dp')"
      - match: |
          final_hidden_states = self.experts(
              hidden_states=hidden_states,
              topk_output=topk_output,
          )
        append: "dumper.dump('moe_expert_output', final_hidden_states, dims='t h[tp:partial] # moe_ep:replicated dp:=attn_dp')"

  # --- DeepEP Normal: dispatch metadata + contiguous GEMM intermediate ---
  - target: sglang.srt.layers.moe.token_dispatcher.deepep._DeepEPDispatcherImplNormal.dispatch_b
    edits:
      - match: "return DeepEPNormalDispatchOutput("
        prepend: |
          dumper.dump('deepep_normal_recv_topk_ids', topk_ids, dims='t_recv[ep] k')
          dumper.dump('deepep_normal_num_recv_tokens_per_expert', num_recv_tokens_per_expert, dims='num_experts')
          dumper.dump('deepep_normal_ep_num_tokens', torch.tensor(hidden_states.shape[0] if not isinstance(hidden_states, tuple) else hidden_states[0].shape[0]))
          dumper.dump('deepep_normal_ep_top_k', torch.tensor(topk_ids.shape[1]))
          dumper.dump('deepep_normal_output_index_ep_num_tokens', torch.tensor(hidden_states.shape[0] if not isinstance(hidden_states, tuple) else hidden_states[0].shape[0]))
          dumper.dump('deepep_normal_output_index_ep_top_k', torch.tensor(topk_ids.shape[1]))

  # --- DeepEP Normal: gate/up GEMM intermediate + src2dst for derouting ---
  # cutlass_w4a8 path (W4A8 models)
  - target: sglang.srt.layers.moe.cutlass_w4a8_moe.cutlass_w4a8_moe_deepep_normal
    edits:
      - match: "silu_and_mul(c1, intermediate)"
        prepend: |
          _n2 = c1.shape[1]
          dumper.dump('gateup_output', c1.view(m * topk, 2, _n2 // 2), dims='t_k[ep] gate_up h_inter # tp:replicated moe_tp:replicated moe_ep:replicated dp:=attn_dp')
          dumper.dump('deepep_normal_src2dst', src2dst)

  # deep_gemm contiguous path (FP8 models using DeepGEMM runner)
  # No dp:=attn_dp here: after EP all-to-all dispatch, both DP ranks have
  # tokens assigned to their local experts (not DP-distributed anymore).
  - target: sglang.srt.layers.moe.moe_runner.deep_gemm.DeepGemmRunnerCore._run_contiguous_gemm
    edits:
      - match: "silu_and_mul(gateup_output.view(-1, N), down_input)"
        prepend: |
          dumper.dump('gateup_output', gateup_output.view(all_tokens, 2, N // 2), dims='t_k[ep] gate_up h_inter # tp:replicated moe_ep:replicated')
          dumper.dump('deepep_normal_output_index', running_state['output_index'])

  # --- DeepEP Low-Latency: dispatch metadata + masked GEMM intermediate ---
  - target: sglang.srt.layers.moe.token_dispatcher.deepep._DeepEPDispatcherImplLowLatency.dispatch_b
    edits:
      - match: "deepep_output = DeepEPLLDispatchOutput("
        prepend: |
          dumper.dump('deepep_ll_masked_m', masked_m, dims='num_experts')
          dumper.dump('deepep_ll_packed_recv_src_info', torch.stack(list(self.handle[0])), dims='num_experts expected_m')
          dumper.dump('deepep_ll_ep_num_tokens', torch.tensor(hidden_states.shape[0] if not isinstance(hidden_states, tuple) else hidden_states[0].shape[0]))
          dumper.dump('deepep_ll_ep_top_k', torch.tensor(topk_ids.shape[1]))

  # --- DeepEP Low-Latency: gate/up GEMM intermediate ---
  # cutlass_w4a8 path (W4A8 models)
  - target: sglang.srt.layers.moe.cutlass_w4a8_moe.cutlass_w4a8_moe_deepep_ll
    edits:
      - match: |
          intermediate_q = torch.empty(
              (num_experts, m, n), device=a.device, dtype=torch.float8_e4m3fn
          )
        prepend: |
          dumper.dump('gateup_output', c1.view(num_experts, m, 2, n), dims='num_experts expected_m gate_up h_inter[ep] # tp:replicated moe_tp:replicated moe_ep:replicated dp:=attn_dp')

  # deep_gemm masked path (FP8 models using DeepGEMM runner)
  # No dp:=attn_dp here: after EP dispatch, both DP ranks have expert data.
  - target: sglang.srt.layers.moe.moe_runner.deep_gemm.DeepGemmRunnerCore._run_masked_gemm
    edits:
      - match: "# Act"
        prepend: |
          dumper.dump('gateup_output', gateup_output.view(num_groups, m, 2, n // 2), dims='num_experts expected_m gate_up h_inter[ep] # tp:replicated moe_ep:replicated')
"""


# ================================= test classes =================================


class TestBF16:
    """E2E tests using BF16 model (Qwen3-30B-A3B) with shared baseline."""

    _baseline_dir: Path

    @classmethod
    def setup_class(cls) -> None:
        cls._baseline_dir = Path(tempfile.mkdtemp(prefix="e2e_baseline_bf16_"))
        config_path: Path = cls._baseline_dir / "patch_config.yaml"
        config_path.write_text(PATCH_CONFIG_YAML)
        _run_server_and_generate(
            model=MODEL_BF16,
            dump_dir=cls._baseline_dir / "dump",
            config_path=config_path,
            tp=BASELINE_TP,
            base_url=DEFAULT_URL_FOR_TEST,
        )
        _verify_patched_fields(
            dump_dir=cls._baseline_dir / "dump",
            field_names=_FIELDS_TO_VERIFY + _FIELDS_GATEUP,
        )

    def test_basic_tp(self, tmp_path: Path) -> None:
        """TP=2 baseline vs TP=4 target."""
        _run_target_and_compare(
            model=MODEL_BF16,
            baseline_exp_dir=self._baseline_dir / "dump" / EXP_NAME,
            tmp_path=tmp_path,
            target_tp=TARGET_TP,
            target_extra_fields=_FIELDS_GATEUP,
        )

    def test_dp_attention(self, tmp_path: Path) -> None:
        """TP=2 baseline vs TP=2+DP=2+dp-attention target.

        In dp-attention mode (attn_tp_size=1, attn_dp_size=2), attention
        tensors are NOT TP-sharded and mlp_output is already all-reduced.
        A separate patch config with corrected dims is used for the target.
        """
        # dp-attention changes the attention computation path (attn_tp_size=1
        # instead of 2), causing BF16 precision differences in attn_output
        # that propagate to pre_mlp_residual -> router_logits -> topk
        # decisions at decision boundaries, even in layer 0.
        _run_target_and_compare(
            model=MODEL_BF16,
            baseline_exp_dir=self._baseline_dir / "dump" / EXP_NAME,
            tmp_path=tmp_path,
            target_tp=BASELINE_TP,
            extra_target_server_args=["--dp", "2", "--enable-dp-attention"],
            target_patch_config_yaml=PATCH_CONFIG_DP_ATTENTION_YAML,
            allow_failed_pattern="gateup_output|moe_topk_ids",
        )

    def test_ep_fused_moe(self, tmp_path: Path) -> None:
        """TP=2 baseline vs TP=4+EP=4 target (FusedMoE StandardDispatcher).

        With --ep-size 4 on TP=4, MoE experts are distributed across all
        4 ranks via FusedMoE/Triton dispatch. Decoder-level tensors remain
        TP-sharded and should compare correctly after unsharding.
        The target uses EP-specific dims with moe_ep:replicated.
        """
        _run_target_and_compare(
            model=MODEL_BF16,
            baseline_exp_dir=self._baseline_dir / "dump" / EXP_NAME,
            tmp_path=tmp_path,
            target_tp=TARGET_TP,
            extra_target_server_args=["--ep-size", "4"],
            target_patch_config_yaml=PATCH_CONFIG_EP_YAML,
            allow_skipped_pattern=_ALLOW_SKIPPED_EP,
            target_extra_fields=_FIELDS_GATEUP,
        )


class TestFP8DeepEP:
    """E2E tests using FP8 model (Qwen3-30B-A3B-FP8) with shared baseline.

    FP8 + TP=4 is incompatible (moe_intermediate_size=768, 768/4=192,
    192 % 128 != 0 -> ValueError in fp8 create_weights). Tests use TP=2.
    BF16 + DeepEP is incompatible (assert False in non-quantized GEMM paths).
    """

    _baseline_dir: Path

    @classmethod
    def setup_class(cls) -> None:
        cls._baseline_dir = Path(tempfile.mkdtemp(prefix="e2e_baseline_fp8_"))
        config_path: Path = cls._baseline_dir / "patch_config.yaml"
        config_path.write_text(PATCH_CONFIG_YAML)
        _run_server_and_generate(
            model=MODEL_FP8,
            dump_dir=cls._baseline_dir / "dump",
            config_path=config_path,
            tp=BASELINE_TP,
            base_url=DEFAULT_URL_FOR_TEST,
        )
        _verify_patched_fields(
            dump_dir=cls._baseline_dir / "dump",
            field_names=_FIELDS_TO_VERIFY + _FIELDS_GATEUP,
        )

    def test_ep_deepep_normal(self, tmp_path: Path) -> None:
        """TP=2 baseline vs TP=2+DeepEP normal target.

        DeepEP normal mode uses all-to-all dispatch with contiguous GEMM.
        --moe-a2a-backend deepep automatically sets ep_size=tp_size=2.
        DeepEP bypasses forward_normal and uses forward_deepep, so MoE
        internals need a separate patch config targeting forward_deepep.
        """
        _run_target_and_compare(
            model=MODEL_FP8,
            baseline_exp_dir=self._baseline_dir / "dump" / EXP_NAME,
            tmp_path=tmp_path,
            target_tp=BASELINE_TP,
            extra_target_server_args=[
                "--dp",
                "2",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
            ],
            target_patch_config_yaml=PATCH_CONFIG_DEEPEP_YAML,
            allow_skipped_pattern=_ALLOW_SKIPPED_DEEPEP,
            target_extra_fields=_FIELDS_GATEUP,
        )

    def test_ep_deepep_low_latency(self, tmp_path: Path) -> None:
        """TP=2 baseline vs TP=2+DeepEP low-latency target.

        DeepEP low-latency mode uses masked GEMM with 3D tensor layout.
        --moe-a2a-backend deepep automatically sets ep_size=tp_size=2.
        DeepEP bypasses forward_normal and uses forward_deepep, so MoE
        internals need a separate patch config targeting forward_deepep.
        """
        _run_target_and_compare(
            model=MODEL_FP8,
            baseline_exp_dir=self._baseline_dir / "dump" / EXP_NAME,
            tmp_path=tmp_path,
            target_tp=BASELINE_TP,
            extra_target_server_args=[
                "--dp",
                "2",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
            ],
            target_patch_config_yaml=PATCH_CONFIG_DEEPEP_YAML,
            allow_skipped_pattern=_ALLOW_SKIPPED_DEEPEP,
            target_extra_fields=_FIELDS_GATEUP,
        )


# ================================== helpers ==================================

_ALLOW_SKIPPED_BASE = ".*"

_ALLOW_SKIPPED_EP = _ALLOW_SKIPPED_BASE

_ALLOW_SKIPPED_DEEPEP = (
    _ALLOW_SKIPPED_BASE
    + "|deepep_normal_recv_topk_ids|deepep_normal_num_recv_tokens_per_expert"
    + "|deepep_normal_ep_num_tokens|deepep_normal_ep_top_k"
    + "|deepep_normal_src2dst"
    + "|deepep_normal_output_index"
    + "|deepep_normal_output_index_ep_num_tokens|deepep_normal_output_index_ep_top_k"
    + "|deepep_ll_masked_m|deepep_ll_packed_recv_src_info"
    + "|deepep_ll_ep_num_tokens|deepep_ll_ep_top_k"
)


def _run_target_and_compare(
    *,
    model: str,
    baseline_exp_dir: Path,
    tmp_path: Path,
    target_tp: int,
    extra_target_server_args: Optional[list[str]] = None,
    target_patch_config_yaml: Optional[str] = None,
    target_extra_fields: Optional[list[str]] = None,
    allow_skipped_pattern: str = _ALLOW_SKIPPED_BASE,
    allow_failed_pattern: Optional[str] = None,
    diff_threshold: Optional[float] = None,
) -> None:
    """Run target server + comparator against a pre-existing baseline."""
    base_url: str = DEFAULT_URL_FOR_TEST

    target_config_path: Path = tmp_path / "patch_config_target.yaml"
    target_config_path.write_text(target_patch_config_yaml or PATCH_CONFIG_YAML)

    target_dir: Path = tmp_path / "target"
    _run_server_and_generate(
        model=model,
        dump_dir=target_dir,
        config_path=target_config_path,
        tp=target_tp,
        base_url=base_url,
        extra_server_args=extra_target_server_args,
    )
    all_fields: list[str] = _FIELDS_TO_VERIFY + (target_extra_fields or [])
    _verify_patched_fields(dump_dir=target_dir, field_names=all_fields)

    target_exp: Path = target_dir / EXP_NAME
    _run_comparator(
        baseline_exp=baseline_exp_dir,
        target_exp=target_exp,
        allow_skipped_pattern=allow_skipped_pattern,
        allow_failed_pattern=allow_failed_pattern,
        diff_threshold=diff_threshold,
    )


def _run_server_and_generate(
    *,
    model: str,
    dump_dir: Path,
    config_path: Path,
    tp: int,
    base_url: str,
    extra_server_args: Optional[list[str]] = None,
) -> None:
    """Launch SGLang server with source patcher + dumper, send a generate request."""
    env: dict[str, str] = {
        **os.environ,
        "DUMPER_SOURCE_PATCHER_CONFIG": str(config_path),
        "DUMPER_DIR": str(dump_dir),
        "DUMPER_EXP_NAME": EXP_NAME,
        "DUMPER_SERVER_PORT": "reuse",
    }

    server_args: list[str] = [
        "--tp",
        str(tp),
        "--max-total-tokens",
        "128",
        "--mem-fraction-static",
        "0.5",
        "--disable-cuda-graph",
        "--disable-radix-cache",
    ]
    if extra_server_args:
        server_args.extend(extra_server_args)

    proc = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=server_args,
        env=env,
    )
    try:
        requests.post(
            f"{base_url}/dumper/configure",
            json={
                "enable": True,
                "filter": DUMPER_FILTER,
                "cleanup_previous": True,
            },
        ).raise_for_status()

        resp = requests.post(
            f"{base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            },
        )
        assert resp.status_code == 200, f"Generate failed: {resp.text}"
    finally:
        kill_process_tree(proc.pid)


def _run_comparator(
    *,
    baseline_exp: Path,
    target_exp: Path,
    extra_args: Optional[list[str]] = None,
    allow_skipped_pattern: str = _ALLOW_SKIPPED_BASE,
    allow_failed_pattern: Optional[str] = None,
    diff_threshold: Optional[float] = None,
) -> None:
    """Run comparator CLI and assert success."""
    cmd: list[str] = [
        "python",
        "-m",
        "sglang.srt.debug_utils.comparator",
        "--baseline-path",
        str(baseline_exp),
        "--target-path",
        str(target_exp),
        "--output-format",
        "json",
        "--allow-skipped-pattern",
        allow_skipped_pattern,
    ]
    if allow_failed_pattern:
        cmd.extend(["--allow-failed-pattern", allow_failed_pattern])
    if diff_threshold is not None:
        cmd.extend(["--diff-threshold", str(diff_threshold)])
    if extra_args:
        cmd.extend(extra_args)

    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    debug_file: Path = _save_comparator_output(
        stdout=result.stdout, stderr=result.stderr
    )
    print(f"Comparator debug output: {debug_file}")

    if result.returncode != 0:
        import json

        failed_names: list[str] = []
        for line in result.stdout.strip().split("\n"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "comparison" and not record.get("passed", True):
                failed_names.append(record.get("name", "<unknown>"))
        print(f"Failed fields: {failed_names}")

    assert (
        result.returncode == 0
    ), f"Comparator failed (rc={result.returncode}). Debug output: {debug_file}"


def _verify_patched_fields(*, dump_dir: Path, field_names: list[str]) -> None:
    """Verify that patched dump fields exist as .pt files."""
    for field in field_names:
        matches: list[Path] = list(dump_dir.rglob(f"*name={field}*.pt"))
        assert len(matches) > 0, (
            f"Expected patched field '{field}' not found under {dump_dir}. "
            f"Available files: {sorted(f.name for f in dump_dir.rglob('*.pt'))[:20]}"
        )


def _save_comparator_output(*, stdout: str, stderr: str) -> Path:
    """Save comparator stdout+stderr to a temp file that persists for debugging."""
    fd, path_str = tempfile.mkstemp(prefix="comparator_e2e_", suffix=".log", dir="/tmp")
    with os.fdopen(fd, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(stdout)
        f.write("\n=== STDERR ===\n")
        f.write(stderr)
    return Path(path_str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
