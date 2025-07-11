import torch
import argparse
from pathlib import Path

import polars as pl


def main(args):
    df_target = read_meta(args.target_path)
    df_target = df_target.sort("forward_pass_id", "dump_index", "rank")
    assert all(c in df_target.columns for c in ["rank", "forward_pass_id", "dump_index", "name"])

    for row in df_target.iter_rows(named=True):
        x_baseline = torch.load(Path(args.baseline_path) / row["filename"], weights_only=True)
        x_target = torch.load(Path(args.target_path) / row["filename"], weights_only=True)
        info = check_tensor_pair(x_baseline=x_baseline, x_target=x_target)
        print(f"Check: {row['filename']}")
        print(TODO)


def read_meta(directory):
    directory = Path(directory)
    assert directory.is_dir()

    rows = []
    for p in directory.glob("*.pt"):
        full_kwargs = {}
        for kv in p.stem.split("___"):
            k, v = kv.split("=")
            full_kwargs[k] = v
        rows.append(
            {
                "filename": str(p.name),
                **full_kwargs,
            }
        )

    return pl.DataFrame(rows)


def check_tensor_pair(x_baseline: torch.Tensor, x_target: torch.Tensor):
    return TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str)
    parser.add_argument("--target-path", type=str)
    args = parser.parse_args()
    main(args)
