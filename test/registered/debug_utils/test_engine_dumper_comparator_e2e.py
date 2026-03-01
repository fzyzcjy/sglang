"""E2E test: source patcher + dumper + comparator on SGLang server.

Patches Qwen3MoeDecoderLayer.forward (and related methods) to insert
dumper.dump() calls at 7 points, launches servers with Qwen3-30B-A3B
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
DUMPER_FILTER = "layer_id in [0, 1, 2]"

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
    "moe_expert_output",
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
        append: "dumper.dump('layer_input', hidden_states, dims='t h # tp:replicated')"
      - match: |
          hidden_states = self.self_attn(
              positions=positions,
              hidden_states=hidden_states,
              forward_batch=forward_batch,
          )
        append: "dumper.dump('attn_output', hidden_states, dims='t h[tp:partial]')"
      - match: |
          hidden_states, residual = self.layer_communicator.prepare_mlp(
              hidden_states, residual, forward_batch
          )
        append: "dumper.dump('pre_mlp_residual', hidden_states, dims='t h # tp:replicated')"
      - match: |
          hidden_states = self.mlp(
              hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
          )
        append: "dumper.dump('mlp_output', hidden_states, dims='t h[tp:partial]')"

  # --- attention internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeAttention.forward_core
    edits:
      - match: "output, _ = self.o_proj(attn_output)"
        prepend: "dumper.dump('attn_pre_o_proj', attn_output, dims='t attn_h[tp]')"

  # --- moe internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward_normal
    edits:
      - match: "router_logits, _ = self.gate(hidden_states)"
        append: "dumper.dump('moe_router_logits', router_logits, dims='t num_experts # tp:replicated')"
      - match: "final_hidden_states = self.experts(hidden_states, topk_output)"
        append: "dumper.dump('moe_expert_output', final_hidden_states, dims='t h[tp:partial]')"
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
        append: "dumper.dump('layer_input', hidden_states, dims='t h # tp:replicated dp:=attn_dp')"
      - match: |
          hidden_states = self.self_attn(
              positions=positions,
              hidden_states=hidden_states,
              forward_batch=forward_batch,
          )
        append: "dumper.dump('attn_output', hidden_states, dims='t h # tp:replicated')"
      - match: |
          hidden_states, residual = self.layer_communicator.prepare_mlp(
              hidden_states, residual, forward_batch
          )
        append: "dumper.dump('pre_mlp_residual', hidden_states, dims='t h # tp:replicated')"
      - match: |
          hidden_states = self.mlp(
              hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
          )
        append: "dumper.dump('mlp_output', hidden_states, dims='t h # tp:replicated')"

  # --- attention internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeAttention.forward_core
    edits:
      - match: "output, _ = self.o_proj(attn_output)"
        prepend: "dumper.dump('attn_pre_o_proj', attn_output, dims='t attn_h # tp:replicated')"

  # --- moe internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward_normal
    edits:
      - match: "router_logits, _ = self.gate(hidden_states)"
        append: "dumper.dump('moe_router_logits', router_logits, dims='t num_experts # tp:replicated')"
      - match: "final_hidden_states = self.experts(hidden_states, topk_output)"
        append: "dumper.dump('moe_expert_output', final_hidden_states, dims='t h[tp:partial]')"
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
        append: "dumper.dump('layer_input', hidden_states, dims='t h # tp:replicated moe_ep:replicated')"
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
      - match: "final_hidden_states = self.experts(hidden_states, topk_output)"
        append: "dumper.dump('moe_expert_output', final_hidden_states, dims='t h[tp:partial] # moe_ep:replicated')"
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
            dump_dir=cls._baseline_dir / "dump", field_names=_FIELDS_TO_VERIFY
        )

    def test_basic_tp(self, tmp_path: Path) -> None:
        """TP=2 baseline vs TP=4 target."""
        _run_target_and_compare(
            model=MODEL_BF16,
            baseline_exp_dir=self._baseline_dir / "dump" / EXP_NAME,
            tmp_path=tmp_path,
            target_tp=TARGET_TP,
        )

    def test_dp_attention(self, tmp_path: Path) -> None:
        """TP=2 baseline vs TP=2+DP=2+dp-attention target.

        In dp-attention mode (attn_tp_size=1, attn_dp_size=2), attention
        tensors are NOT TP-sharded and mlp_output is already all-reduced.
        A separate patch config with corrected dims is used for the target.
        """
        _run_target_and_compare(
            model=MODEL_BF16,
            baseline_exp_dir=self._baseline_dir / "dump" / EXP_NAME,
            tmp_path=tmp_path,
            target_tp=BASELINE_TP,
            extra_target_server_args=["--dp", "2", "--enable-dp-attention"],
            target_patch_config_yaml=PATCH_CONFIG_DP_ATTENTION_YAML,
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
            dump_dir=cls._baseline_dir / "dump", field_names=_FIELDS_TO_VERIFY
        )

    def test_ep_deepep_normal(self, tmp_path: Path) -> None:
        """TP=2 baseline vs TP=2+DeepEP normal target.

        DeepEP normal mode uses all-to-all dispatch with contiguous GEMM.
        --moe-a2a-backend deepep automatically sets ep_size=tp_size=2.
        """
        _run_target_and_compare(
            model=MODEL_FP8,
            baseline_exp_dir=self._baseline_dir / "dump" / EXP_NAME,
            tmp_path=tmp_path,
            target_tp=BASELINE_TP,
            extra_target_server_args=[
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
            ],
            target_patch_config_yaml=PATCH_CONFIG_EP_YAML,
        )

    def test_ep_deepep_low_latency(self, tmp_path: Path) -> None:
        """TP=2 baseline vs TP=2+DeepEP low-latency target.

        DeepEP low-latency mode uses masked GEMM with 3D tensor layout.
        --moe-a2a-backend deepep automatically sets ep_size=tp_size=2.
        """
        _run_target_and_compare(
            model=MODEL_FP8,
            baseline_exp_dir=self._baseline_dir / "dump" / EXP_NAME,
            tmp_path=tmp_path,
            target_tp=BASELINE_TP,
            extra_target_server_args=[
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
            ],
            target_patch_config_yaml=PATCH_CONFIG_EP_YAML,
        )


# ================================== helpers ==================================


def _run_target_and_compare(
    *,
    model: str,
    baseline_exp_dir: Path,
    tmp_path: Path,
    target_tp: int,
    extra_target_server_args: Optional[list[str]] = None,
    target_patch_config_yaml: Optional[str] = None,
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
    _verify_patched_fields(dump_dir=target_dir, field_names=_FIELDS_TO_VERIFY)

    target_exp: Path = target_dir / EXP_NAME
    _run_comparator(baseline_exp=baseline_exp_dir, target_exp=target_exp)


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


def _run_comparator(*, baseline_exp: Path, target_exp: Path) -> None:
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
        "input_ids|positions",
    ]

    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    debug_file: Path = _save_comparator_output(
        stdout=result.stdout, stderr=result.stderr
    )
    print(f"Comparator debug output: {debug_file}")

    assert result.returncode == 0, (
        f"Comparator failed (rc={result.returncode}). Debug output: {debug_file}"
    )


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
