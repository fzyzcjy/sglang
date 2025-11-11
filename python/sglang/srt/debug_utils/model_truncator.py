# This file also references Slime :: fp8_cast_bf16.py

import torch
import json
import os
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict

from safetensors.torch import load_file, save_file


def main(args):
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    for pattern in ["config.json", "generation_config.json", "*.py", "tokenizer*"]:
        os.system(f"cp -rf {input_path}/{pattern} {output_path}")

    filename_index = "model.safetensors.index.json"
    safetensors_index = json.loads((input_path / filename_index).read_text())
    _transform_index(safetensors_index)
    (output_path / filename_index).write_text(json.dumps(safetensors_index, indent=4))

    for path_safetensor in sorted(list(input_path.glob("*.safetensors"))):
        state_dict = load_file(path_safetensor)
        _transform_safetensor_file(state_dict, safetensors_index)
        save_file(state_dict, output_path / path_safetensor.relative_to(input_path))


def _transform_index(safetensors_index):
    TODO


def _transform_safetensor_file(state_dict: Dict[str, torch.Tensor], safetensors_index: Dict[str, str]):
    TODO


if __name__ == "__main__":
    parser = ArgumentParser(description="Create truncated model for fast debugging.")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    main(parser.parse_args())
