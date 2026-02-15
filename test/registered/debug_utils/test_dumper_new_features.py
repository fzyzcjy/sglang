"""Tests for new dumper features (lazy value, output dict mode, static metadata, grad dump, param grads).

This file imports dumper directly via sys.path to avoid the heavy sglang import chain.
"""

import sys
from pathlib import Path

import pytest
import torch

# Import dumper standalone (it's designed to work without sglang package)
_DUMPER_DIR = str(
    Path(__file__).resolve().parents[3] / "python" / "sglang" / "srt" / "debug_utils"
)
sys.path.insert(0, _DUMPER_DIR)
import dumper as dumper_module  # noqa: E402

_Dumper = dumper_module._Dumper


# -------------------------------------- test helpers ------------------------------------------


def _make_test_dumper(tmp_path: Path, **overrides) -> "_Dumper":
    """Create a _Dumper for CPU testing without HTTP server or distributed."""
    d = _Dumper()
    d._enable = True
    d._base_dir = tmp_path
    d._partial_name = "test"
    d._http_server_handled = True
    d._forward_pass_id = 1
    for key, value in overrides.items():
        setattr(d, f"_{key}", value)
    return d


def _get_filenames(tmpdir) -> set:
    return {f.name for f in Path(tmpdir).glob("sglang_dump_*/*.pt")}


def _find_dump_file(tmpdir, *, rank: int, name: str) -> Path:
    matches = [
        f
        for f in Path(tmpdir).glob("sglang_dump_*/*.pt")
        if f"rank={rank}" in f.name and name in f.name
    ]
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching rank={rank} name={name}, got {matches}"
    return matches[0]


# -------------------------------------- Task 1: Lazy value ------------------------------------------


class TestLazyValue:
    def test_materialize_value_callable(self):
        tensor = torch.randn(3, 3)
        result = dumper_module._materialize_value(lambda: tensor)
        assert torch.equal(result, tensor)

    def test_materialize_value_passthrough(self):
        tensor = torch.randn(3, 3)
        result = dumper_module._materialize_value(tensor)
        assert result is tensor

    def test_deepcopy_or_clone_tensor(self):
        tensor = torch.randn(3, 3)
        cloned = dumper_module._deepcopy_or_clone(tensor)
        assert torch.equal(cloned, tensor)
        assert cloned is not tensor
        assert cloned.data_ptr() != tensor.data_ptr()

    def test_deepcopy_or_clone_dict(self):
        original = {"a": [1, 2, 3]}
        copied = dumper_module._deepcopy_or_clone(original)
        assert copied == original
        assert copied is not original
        assert copied["a"] is not original["a"]

    def test_dump_with_callable_value(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        tensor = torch.randn(4, 4)
        d.dump("lazy_tensor", lambda: tensor)

        filenames = _get_filenames(tmp_path)
        assert any("name=lazy_tensor" in f for f in filenames)

        path = _find_dump_file(tmp_path, rank=0, name="lazy_tensor")
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert torch.equal(loaded, tensor)


# -------------------------------------- Task 2: Output Dict Mode ------------------------------------------


class TestOutputDictMode:
    def test_save_value_normal_mode(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        d._output_dict_mode = False
        tensor = torch.randn(3, 3)
        path = str(tmp_path / "normal.pt")

        d._save_value(tensor, path, {"name": "test"})

        loaded = torch.load(path, weights_only=True)
        assert torch.equal(loaded, tensor)

    def test_save_value_dict_mode(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        d._output_dict_mode = True
        tensor = torch.randn(3, 3)
        path = str(tmp_path / "dict.pt")

        d._save_value(tensor, path, {"name": "test"})

        loaded = torch.load(path, weights_only=False, map_location="cpu")
        assert isinstance(loaded, dict)
        assert "value" in loaded
        assert "meta" in loaded
        assert torch.equal(loaded["value"], tensor)
        assert loaded["meta"]["name"] == "test"

    def test_dump_output_dict_mode_integration(self, tmp_path):
        d = _make_test_dumper(tmp_path, output_dict_mode=True)
        tensor = torch.randn(4, 4)

        d.dump("dict_test", tensor)

        path = _find_dump_file(tmp_path, rank=0, name="dict_test")
        loaded = torch.load(path, weights_only=False, map_location="cpu")
        assert isinstance(loaded, dict)
        assert torch.equal(loaded["value"], tensor)
        assert loaded["meta"]["name"] == "dict_test"
        assert loaded["meta"]["rank"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
