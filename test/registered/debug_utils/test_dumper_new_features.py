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


# -------------------------------------- Task 3: Static Metadata ------------------------------------------


class TestStaticMetadata:
    def test_static_meta_contains_world_info(self):
        d = _make_test_dumper(Path("/tmp"))
        meta = d._get_static_meta()
        assert "world_rank" in meta
        assert "world_size" in meta
        assert meta["world_rank"] == 0
        assert meta["world_size"] == 1

    def test_static_meta_caching(self):
        d = _make_test_dumper(Path("/tmp"))
        meta1 = d._get_static_meta()
        meta2 = d._get_static_meta()
        assert meta1 is meta2

    def test_parallel_info_graceful_fallback(self):
        sglang_info = dumper_module._collect_sglang_parallel_info()
        assert isinstance(sglang_info, dict)

        megatron_info = dumper_module._collect_megatron_parallel_info()
        assert isinstance(megatron_info, dict)

    def test_dict_mode_includes_static_meta(self, tmp_path):
        d = _make_test_dumper(tmp_path, output_dict_mode=True)
        tensor = torch.randn(2, 2)

        d.dump("meta_test", tensor)

        path = _find_dump_file(tmp_path, rank=0, name="meta_test")
        loaded = torch.load(path, weights_only=False, map_location="cpu")
        meta = loaded["meta"]
        assert "world_rank" in meta
        assert "world_size" in meta


# -------------------------------------- Task 4: dump grad ------------------------------------------


class TestDumpGrad:
    def test_dump_grad_basic(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        x = torch.randn(3, 3, requires_grad=True)
        y = (x * 2).sum()

        d.dump("test_tensor", x)
        y.backward()

        filenames = _get_filenames(tmp_path)
        assert any("name=test_tensor" in f and "grad__" not in f for f in filenames)
        assert any("grad__test_tensor" in f for f in filenames)

    def test_dump_grad_non_tensor_skipped(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        d.dump("not_tensor", 42)

        filenames = _get_filenames(tmp_path)
        assert not any("grad__" in f for f in filenames)

    def test_dump_grad_no_requires_grad_skipped(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        x = torch.randn(3, 3, requires_grad=False)
        d.dump("no_grad_tensor", x)

        filenames = _get_filenames(tmp_path)
        assert any("name=no_grad_tensor" in f for f in filenames)
        assert not any("grad__" in f for f in filenames)

    def test_dump_grad_captures_forward_pass_id(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        d._forward_pass_id = 42
        x = torch.randn(3, 3, requires_grad=True)
        y = (x * 2).sum()

        d.dump("id_test", x)
        d._forward_pass_id = 999
        y.backward()

        grad_files = [
            f.name
            for f in tmp_path.glob("sglang_dump_*/*.pt")
            if "grad__" in f.name
        ]
        assert len(grad_files) == 1
        assert "forward_pass_id=42" in grad_files[0]

    def test_dump_grad_file_content(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = (x * 3).sum()

        d.dump("content_check", x)
        y.backward()

        grad_path = [
            f
            for f in tmp_path.glob("sglang_dump_*/*.pt")
            if "grad__content_check" in f.name
        ][0]
        loaded = torch.load(grad_path, map_location="cpu", weights_only=True)
        expected_grad = torch.full((2, 2), 3.0)
        assert torch.equal(loaded, expected_grad)

    def test_disable_forward_dump(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_forward_dump=False)
        x = torch.randn(3, 3, requires_grad=True)
        y = (x * 2).sum()

        d.dump("fwd_disabled", x)
        y.backward()

        filenames = _get_filenames(tmp_path)
        assert not any(
            "name=fwd_disabled" in f and "grad__" not in f for f in filenames
        )
        assert any("grad__fwd_disabled" in f for f in filenames)

    def test_disable_grad_dump(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_grad_dump=False)
        x = torch.randn(3, 3, requires_grad=True)
        y = (x * 2).sum()

        d.dump("grad_disabled", x)
        y.backward()

        filenames = _get_filenames(tmp_path)
        assert any("name=grad_disabled" in f for f in filenames)
        assert not any("grad__" in f for f in filenames)

    def test_dump_grad_no_none_format_in_filename(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        x = torch.randn(3, 3, requires_grad=True)
        y = (x * 2).sum()

        d.dump("no_none_test", x)
        y.backward()

        grad_files = [
            f.name
            for f in tmp_path.glob("sglang_dump_*/*.pt")
            if "grad__" in f.name
        ]
        assert len(grad_files) == 1
        assert "format=None" not in grad_files[0]
        assert "cp_mode=None" not in grad_files[0]

    def test_dump_format_and_cp_mode_in_filename(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        tensor = torch.randn(4, 4)

        d.dump("formatted", tensor, format="bshd", cp_mode="zigzag")

        filenames = _get_filenames(tmp_path)
        matching = [f for f in filenames if "name=formatted" in f]
        assert len(matching) == 1
        assert "format=bshd" in matching[0]
        assert "cp_mode=zigzag" in matching[0]


# -------------------------------------- Task 5: dump param grads ------------------------------------------


class TestDumpParamGrads:
    def test_basic(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        model = torch.nn.Linear(4, 2)
        x = torch.randn(3, 4)
        y = model(x).sum()
        y.backward()

        d.dump_param_grads(model, name_prefix="model")

        filenames = _get_filenames(tmp_path)
        assert any("model_grad__weight" in f for f in filenames)
        assert any("model_grad__bias" in f for f in filenames)

    def test_no_grad_skipped(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        model = torch.nn.Linear(4, 2)

        d.dump_param_grads(model, name_prefix="model")

        filenames = _get_filenames(tmp_path)
        assert len(filenames) == 0

    def test_filter(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        d._filter = "weight"
        model = torch.nn.Linear(4, 2)
        x = torch.randn(3, 4)
        y = model(x).sum()
        y.backward()

        d.dump_param_grads(model, name_prefix="model")

        filenames = _get_filenames(tmp_path)
        assert any("model_grad__weight" in f for f in filenames)
        assert not any("model_grad__bias" in f for f in filenames)

    def test_file_content(self, tmp_path):
        d = _make_test_dumper(tmp_path)
        model = torch.nn.Linear(4, 2, bias=False)
        x = torch.ones(1, 4)
        y = model(x).sum()
        y.backward()

        d.dump_param_grads(model, name_prefix="p")

        path = [
            f
            for f in tmp_path.glob("sglang_dump_*/*.pt")
            if "p_grad__weight" in f.name
        ][0]
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert torch.equal(loaded, model.weight.grad)

    def test_disabled(self, tmp_path):
        d = _make_test_dumper(tmp_path, enable_grad_dump=False)
        model = torch.nn.Linear(4, 2)
        x = torch.randn(3, 4)
        y = model(x).sum()
        y.backward()

        d.dump_param_grads(model, name_prefix="model")

        filenames = _get_filenames(tmp_path)
        assert len(filenames) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
