import torch
import argparse
from pathlib import Path

import polars as pl


def main(args):
    df_target = read_meta(args.target_path)
    df_target = df_target.sort("forward_pass_id", "dump_index", "rank")
    assert all(c in df_target.columns for c in ["rank", "forward_pass_id", "dump_index", "name"])

    for row in df_target.iter_rows(named=True):
        print()
        print(f"Check: {row['filename']}")
        check_tensor_pair(
            path_baseline=Path(args.baseline_path) / row["filename"],
            path_target=Path(args.target_path) / row["filename"],
        )


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


def check_tensor_pair(path_baseline, path_target):
    x_baseline = torch.load(path_baseline, weights_only=True)
    x_target = torch.load(path_target, weights_only=True)

    print(
        f"[shape] {x_baseline.shape} vs {x_target.shape}\t"
        f"[dtype] {x_baseline.dtype} vs {x_target.dtype}"
    )

    if x_baseline.shape != x_target.shape:
        print(f"❌ Shape mismatch")
        return

    abs_diff = (x_target - x_baseline).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str)
    parser.add_argument("--target-path", type=str)
    args = parser.parse_args()
    main(args)
