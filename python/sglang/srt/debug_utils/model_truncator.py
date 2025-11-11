# This file also references Slime :: fp8_cast_bf16.py
import re

import torch
import json
import os
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict

from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download


def main(args):
    input_path = Path(snapshot_download(args.input))
    output_path = Path(args.output)
    print(f"{input_path=} {output_path=}")

    output_path.mkdir(parents=True, exist_ok=True)

    for pattern in ["config.json", "generation_config.json", "*.py", "tokenizer*"]:
        os.system(f"cp -rf {input_path}/{pattern} {output_path}")

    filename_index = "model.safetensors.index.json"
    safetensors_index = json.loads((input_path / filename_index).read_text())
    _transform_index(args, safetensors_index)
    (output_path / filename_index).write_text(json.dumps(safetensors_index, indent=4))

    for path_input_safetensors in sorted(list(input_path.glob("*.safetensors"))):
        path_output_safetensors = output_path / path_input_safetensors.relative_to(input_path)

        state_dict = load_file(path_input_safetensors)
        _transform_safetensors_file(state_dict, safetensors_index, debug_name=str(path_output_safetensors))
        if len(state_dict) > 0:
            print(f"Save {len(state_dict)} tensors to {path_output_safetensors}")
            save_file(state_dict, path_output_safetensors)
        else:
            print(f"Skip saving {path_output_safetensors} since it is empty")


def _transform_index(args, safetensors_index):
    weight_map = safetensors_index["weight_map"]
    weight_map = {name: loc for name, loc in weight_map.items() if _filter_tensor_name(args, name)}
    safetensors_index["weight_map"] = weight_map


def _transform_safetensors_file(state_dict: Dict[str, torch.Tensor], safetensors_index, debug_name: str):
    names_to_remove = set(state_dict) - set(safetensors_index["weight_map"])
    print(f"Remove {names_to_remove} in {debug_name}")
    for name in names_to_remove:
        del state_dict[name]


def _filter_tensor_name(args, tensor_name: str):
    # We focus on DeepSeek-like names currently, but can be easily extended to more kinds of models
    m = re.match(r"^model.layers.(\d+).*", tensor_name)
    if m is None:
        return True

    layer_id = int(m.group(0))
    return layer_id < args.keep_num_layers


if __name__ == "__main__":
    """
    Example:
    python sglang.srt.debug_utils.model_truncator --input deepseek-ai/DeepSeek-V3-0324 --output /tmp/DeepSeek-V3-0324-5layer
    hf upload my_name/DeepSeek-V3-0324-5layer /tmp/DeepSeek-V3-0324-5layer
    """
    parser = ArgumentParser(description="Create truncated model for fast debugging.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--keep-num-layers", type=int, default=5)
    main(parser.parse_args())
