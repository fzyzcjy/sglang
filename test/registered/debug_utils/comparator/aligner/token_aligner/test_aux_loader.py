import sys

import pytest

from sglang.srt.debug_utils.comparator.aligner.token_aligner.aux_loader import (
    _ensure_dims_in_metas,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.aux_plugins import (
    _MegatronPlugin,
    _SGLangPlugin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)

_sglang_plugin = _SGLangPlugin()
_megatron_plugin = _MegatronPlugin()


class TestEnsureDimsInMetas:
    """Tests for _ensure_dims_in_metas."""

    def _make_meta(self, *, cp_size: int = 1, cp_rank: int = 0) -> dict:
        return {
            "sglang_parallel_info": {
                "tp_rank": 0,
                "tp_size": 1,
                "cp_rank": cp_rank,
                "cp_size": cp_size,
            }
        }

    def test_no_cp_returns_metas_unchanged(self):
        """Without CP parallelism, metas are returned as-is."""
        metas: list[dict] = [self._make_meta(cp_size=1)]
        result = _ensure_dims_in_metas(name="input_ids", plugin=_sglang_plugin, metas=metas)
        assert result is metas

    def test_dims_already_present_returns_metas_unchanged(self):
        """If dims is already in meta, metas are returned as-is."""
        metas: list[dict] = [{**self._make_meta(cp_size=2, cp_rank=0), "dims": "t"}]
        result = _ensure_dims_in_metas(name="input_ids", plugin=_sglang_plugin, metas=metas)
        assert result is metas

    def test_cp_sharded_sglang_input_ids_raises(self):
        """CP + input_ids in sglang raises NotImplementedError."""
        metas: list[dict] = [
            self._make_meta(cp_size=2, cp_rank=0),
            self._make_meta(cp_size=2, cp_rank=1),
        ]
        with pytest.raises(NotImplementedError, match="CP-sharded"):
            _ensure_dims_in_metas(name="input_ids", plugin=_sglang_plugin, metas=metas)

    def test_cp_sharded_sglang_positions_raises(self):
        """CP + positions in sglang raises NotImplementedError."""
        metas: list[dict] = [
            self._make_meta(cp_size=2, cp_rank=0),
            self._make_meta(cp_size=2, cp_rank=1),
        ]
        with pytest.raises(NotImplementedError, match="CP-sharded"):
            _ensure_dims_in_metas(name="positions", plugin=_sglang_plugin, metas=metas)

    def test_cp_sharded_megatron_input_ids_raises(self):
        """CP + input_ids in megatron raises NotImplementedError."""
        metas: list[dict] = [
            {"megatron_parallel_info": {"cp_rank": 0, "cp_size": 2}},
            {"megatron_parallel_info": {"cp_rank": 1, "cp_size": 2}},
        ]
        with pytest.raises(NotImplementedError, match="CP-sharded"):
            _ensure_dims_in_metas(name="input_ids", plugin=_megatron_plugin, metas=metas)

    def test_cp_non_sharded_name_returns_metas_unchanged(self):
        """CP + non-sharded tensor name (seq_lens) returns metas as-is."""
        metas: list[dict] = [
            self._make_meta(cp_size=2, cp_rank=0),
            self._make_meta(cp_size=2, cp_rank=1),
        ]
        result = _ensure_dims_in_metas(name="seq_lens", plugin=_sglang_plugin, metas=metas)
        assert result is metas

    def test_unknown_plugin_returns_metas_unchanged(self):
        """CP + plugin with empty cp_sharded_names returns metas as-is."""

        class _DummyPlugin(_SGLangPlugin):
            @property
            def cp_sharded_names(self) -> frozenset[str]:
                return frozenset()

        metas: list[dict] = [
            self._make_meta(cp_size=2, cp_rank=0),
            self._make_meta(cp_size=2, cp_rank=1),
        ]
        result = _ensure_dims_in_metas(name="input_ids", plugin=_DummyPlugin(), metas=metas)
        assert result is metas


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
