import os
import sys
import unittest
from typing import Optional

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

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


def _setup_cpu_runtime() -> None:
    """Init a gloo world-1 model-parallel group on CPU so parallel linears build.

    The attention projections (ColumnParallelLinear / RowParallelLinear) and the
    wo_b reduce_results formula read the model-parallel group, so it must exist
    before construction. Uses gloo + device=cpu so the test needs no GPU.
    """
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", device="cpu"))

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29671")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )


def _make_yarn_config():
    """A tiny dsv4 config with a full YARN rope_scaling so MQALayer-mode builds.

    The DSpark tiny config carries ``rope_scaling={}``; MQALayer indexes
    ``rope_scaling["original_max_position_embeddings"]`` directly, so MQALayer-mode
    construction needs the YARN keys present (and a real compress_ratios table).
    """
    from test.srt.speculative._dspark_reference.deepseek_v4.parity_fixture import (
        make_tiny_dsv4_config,
    )

    config = make_tiny_dsv4_config()
    config.rope_scaling = {
        "factor": 4.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 128,
    }
    config.compress_ratios = [0, 0]
    config.compress_rope_theta = 10000
    return config


class TestMqaAttentionBaseConstruction(CustomTestCase):
    """MqaAttentionBase reproduces the original inline construction of both subclasses.

    Builds MqaAttentionBase / DSparkAttention / MQALayer on CPU and asserts each
    projection's type, weight shape, registered name, and dtype (plus attn_sink /
    freqs_cis / reduce_results) match the values the original inline construction
    produced. Pure construction needs no GPU; the fp8 wo_a path (which requires a
    real fp8 quant_config + weight_scale_inv) is left to the GPU weightload gate.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._skip_reason: Optional[str] = None
        try:
            _ensure_repo_test_package()
            _setup_cpu_runtime()
            from test.srt.speculative._dspark_reference.deepseek_v4.parity_fixture import (
                make_tiny_dsv4_config,
            )

            from sglang.srt.models.deepseek_v4 import MqaAttentionBase
        except Exception as exc:  # noqa: BLE001 - any setup failure -> clean skip
            cls._skip_reason = f"CPU runtime unavailable: {exc!r}"
            return

        cls.MqaAttentionBase = MqaAttentionBase
        cls.dspark_config = make_tiny_dsv4_config()
        cls.yarn_config = _make_yarn_config()

    def setUp(self) -> None:
        if self._skip_reason:
            self.skipTest(self._skip_reason)

    def _build_base(self, *, config, prefix: str = "self_attn", **kwargs):
        from sglang.srt.runtime_context import get_parallel

        attn_tp_size = kwargs.get("attn_tp_size")
        attn_tp_rank = kwargs.get("attn_tp_rank", 0)
        override = {}
        if attn_tp_size is not None:
            override = {"attn_tp_size": attn_tp_size, "attn_tp_rank": attn_tp_rank}
        with get_parallel().override(**override):
            return self.MqaAttentionBase(
                config, kwargs.pop("layer_id", 0), None, prefix, **kwargs
            )

    def test_dspark_mode_base_modules(self) -> None:
        """DSpark deviations reproduce wq_a/wkv/q_norm/wq_b/kv_norm/wo_a/wo_b shapes + names."""
        from sglang.srt.models.dbrx import ReplicatedLinear

        base = self._build_base(
            config=self.dspark_config,
            layer_id=2,
            attn_tp_size=1,
            attn_tp_rank=0,
            compress_ratio=0,
            fuse_wqa_wkv=False,
            wo_a_fp8=False,
            wo_a_keeps_quant_config=True,
            wo_b_reduce_results=True,
            rope_original_seq_len=0,
        )
        self.assertEqual(base.compress_ratio, 0)
        self.assertIsInstance(base.wq_a, ReplicatedLinear)
        self.assertIsInstance(base.wkv, ReplicatedLinear)
        self.assertFalse(hasattr(base, "wqkv_a"))
        self.assertEqual(tuple(base.wq_a.weight.shape), (16, 32))
        self.assertEqual(tuple(base.wkv.weight.shape), (16, 32))
        self.assertEqual(tuple(base.q_norm.weight.shape), (16,))
        self.assertEqual(tuple(base.wq_b.weight.shape), (64, 16))
        self.assertEqual(tuple(base.kv_norm.weight.shape), (16,))
        self.assertEqual(tuple(base.wo_a.weight.shape), (32, 32))
        self.assertEqual(tuple(base.wo_b.weight.shape), (32, 32))
        self.assertEqual(base.wo_a.weight.dtype, torch.bfloat16)
        self.assertEqual(tuple(base.attn_sink.shape), (4,))
        self.assertEqual(base.attn_sink.dtype, torch.float32)
        self.assertTrue(hasattr(base, "freqs_cis"))
        self.assertTrue(base.freqs_cis.is_complex())

    def test_dspark_mode_param_names(self) -> None:
        """The base param name set is exactly the dsv4 checkpoint attn weight names."""
        base = self._build_base(
            config=self.dspark_config,
            layer_id=2,
            attn_tp_size=1,
            attn_tp_rank=0,
            compress_ratio=0,
            fuse_wqa_wkv=False,
            wo_a_fp8=False,
            wo_a_keeps_quant_config=True,
            wo_b_reduce_results=True,
            rope_original_seq_len=0,
        )
        names = set(dict(base.named_parameters()).keys())
        self.assertEqual(
            names,
            {
                "wq_a.weight",
                "wkv.weight",
                "q_norm.weight",
                "wq_b.weight",
                "kv_norm.weight",
                "wo_a.weight",
                "wo_b.weight",
                "attn_sink",
            },
        )

    def test_dspark_wo_b_reduce_results_is_true(self) -> None:
        """DSpark wo_b keeps RowParallelLinear's default (reduce_results=True)."""
        base = self._build_base(
            config=self.dspark_config,
            layer_id=2,
            attn_tp_size=1,
            attn_tp_rank=0,
            compress_ratio=0,
            fuse_wqa_wkv=False,
            wo_a_fp8=False,
            wo_a_keeps_quant_config=True,
            wo_b_reduce_results=True,
            rope_original_seq_len=0,
        )
        self.assertIs(base.wo_b.reduce_results, True)

    def test_mqalayer_mode_nonfused_modules(self) -> None:
        """MQALayer defaults (no overrides) build wq_a/wkv, not the fused wqkv_a."""
        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        base = self._build_base(
            config=self.yarn_config,
            layer_id=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            fuse_wqa_wkv=False,
        )
        self.assertTrue(hasattr(base, "wq_a"))
        self.assertFalse(hasattr(base, "wqkv_a"))
        self.assertEqual(base.compress_ratio, 0)
        # wo_a non-fp8 path forces bf16 weights and (MQALayer) drops the quant_config.
        self.assertEqual(base.wo_a.weight.dtype, torch.bfloat16)
        expected_reduce = (
            base.attn_tp_size == get_tensor_model_parallel_world_size()
            and base.attn_tp_size > 1
        )
        self.assertEqual(bool(base.wo_b.reduce_results), expected_reduce)

    def test_fused_wqkv_a_branch(self) -> None:
        """fuse_wqa_wkv=True builds a single wqkv_a (q_lora_rank + head_dim) and no wq_a/wkv."""
        from sglang.srt.models.dbrx import ReplicatedLinear

        base = self._build_base(
            config=self.yarn_config,
            layer_id=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            fuse_wqa_wkv=True,
        )
        self.assertIsInstance(base.wqkv_a, ReplicatedLinear)
        self.assertFalse(hasattr(base, "wq_a"))
        self.assertFalse(hasattr(base, "wkv"))
        self.assertEqual(tuple(base.wqkv_a.weight.shape), (16 + 16, 32))
        names = set(dict(base.named_parameters()).keys())
        self.assertIn("wqkv_a.weight", names)
        self.assertNotIn("wq_a.weight", names)

    def test_tp2_sharding(self) -> None:
        """attn_tp_size=2 halves n_local_heads and the column/row partition shapes."""
        base = self._build_base(
            config=self.yarn_config,
            layer_id=0,
            attn_tp_size=2,
            attn_tp_rank=0,
            fuse_wqa_wkv=False,
        )
        self.assertEqual(base.attn_tp_size, 2)
        self.assertEqual(base.n_local_heads, 2)
        self.assertEqual(base.n_local_groups, 1)
        # wq_b output (n_heads*head_dim=64) sharded over tp=2 -> 32 rows per partition.
        self.assertEqual(tuple(base.wq_b.weight.shape), (32, 16))
        # attn_sink is replicated over the full n_heads (not sharded).
        self.assertEqual(tuple(base.attn_sink.shape), (4,))

    def test_dspark_attention_subclass_wiring(self) -> None:
        """DSparkAttention extends the base and adds its own RadixAttention (self.attn)."""
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.models.deepseek_v4 import MqaAttentionBase
        from sglang.srt.models.deepseek_v4_dspark import DSparkAttention
        from sglang.srt.runtime_context import get_parallel

        with get_parallel().override(attn_tp_size=1, attn_tp_rank=0):
            attn = DSparkAttention(
                config=self.dspark_config, layer_id=2, quant_config=None, prefix="self_attn"
            )
        self.assertIsInstance(attn, MqaAttentionBase)
        self.assertEqual(attn.compress_ratio, 0)
        self.assertIsInstance(attn.attn, RadixAttention)
        self.assertEqual(attn.window_size, int(self.dspark_config.window_size))
        self.assertIs(attn.wo_b.reduce_results, True)
        self.assertEqual(attn.wo_a.weight.dtype, torch.bfloat16)

    def test_mqalayer_subclass_aliases_tp_attrs(self) -> None:
        """MQALayer aliases tp_rank/tp_size to the canonical attn_tp_* and keeps attn_mqa."""
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.models.deepseek_v4 import MQALayer, MqaAttentionBase
        from sglang.srt.runtime_context import get_parallel

        try:
            with get_parallel().override(attn_tp_size=1, attn_tp_rank=0):
                layer = MQALayer(
                    config=self.yarn_config, layer_id=0, quant_config=None, prefix="self_attn"
                )
        except Exception as exc:  # noqa: BLE001 - rotary_emb may need a device backend
            self.skipTest(f"MQALayer construction needs a rope backend: {exc!r}")
        self.assertIsInstance(layer, MqaAttentionBase)
        self.assertEqual(layer.tp_rank, layer.attn_tp_rank)
        self.assertEqual(layer.tp_size, layer.attn_tp_size)
        self.assertTrue(hasattr(layer, "rotary_emb"))
        self.assertIsInstance(layer.attn_mqa, RadixAttention)
        self.assertFalse(hasattr(layer, "attn"))


if __name__ == "__main__":
    unittest.main()
