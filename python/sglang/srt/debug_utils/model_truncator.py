# This file also references Slime :: fp8_cast_bf16.py

import os
from pathlib import Path
from argparse import ArgumentParser


def main(args):
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    for pattern in ["config.json", "*.py", "tokenizer*"]:
        os.system(f"cp -rf {input_path}/{pattern} {output_path}")

    TODO


if __name__ == "__main__":
    parser = ArgumentParser(description="Create truncated model for fast debugging.")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    main(parser.parse_args())
