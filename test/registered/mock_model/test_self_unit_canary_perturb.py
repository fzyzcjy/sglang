import logging
from unittest.mock import Mock

from _pytest.logging import LogCaptureFixture

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.perturb import real_kv_used
from sglang.srt.kv_canary.perturb.config import PerturbConfig, TargetGroupKind
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_real_kv_used_logs_when_target_group_has_no_real_kv_sources(
    caplog: LogCaptureFixture,
) -> None:
    config = PerturbConfig(
        req_to_token_prob=0.0,
        real_kv_used_prob=1.0,
        real_kv_unused_cache_prob=0.0,
        target_group_kind=TargetGroupKind.FULL,
        warmup_steps=0,
    )
    warmup_gate = Mock()
    warmup_gate.is_in_warmup.return_value = False
    group = CanaryBufferGroup(
        kind=PoolKind.FULL,
        k_head=Mock(),
        k_tail=Mock(),
        v_head=None,
        v_tail=None,
        real_kv_sources_k=(),
        real_kv_sources_v=(),
        swa_index_lut=None,
    )

    with caplog.at_level(logging.INFO, logger=real_kv_used.logger.name):
        real_kv_used.run(
            forward_batch=Mock(),
            config=config,
            req_to_token_pool=Mock(),
            buffer_groups=(group,),
            swa_window_size=0,
            warmup_gate=warmup_gate,
        )

    assert (
        "kv_canary perturb real_kv_used: skipped because no target group with "
        "real_kv_sources_k matched target_group_kind=full"
    ) in caplog.text
