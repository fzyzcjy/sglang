import argparse
import re
from pathlib import Path

import polars as pl
import torch

from sglang.srt.debug_utils.dumper import get_truncated_value


def main(args):
    df_target = read_meta(args.target_path)
    df_target = df_target.sort("rank", "dump_index")
    df_target = df_target.filter(
        (pl.col("forward_pass_id") >= args.start_id)
        & (pl.col("forward_pass_id") <= args.end_id)
    )
    assert all(
        c in df_target.columns
        for c in ["rank", "forward_pass_id", "dump_index", "name"]
    )

    for row in df_target.iter_rows(named=True):
        baseline_id = row["forward_pass_id"] - args.start_id + args.baseline_start_id
        baseline_filename = re.sub(r"forward_pass_id=(\d+)", f"forward_pass_id={baseline_id}", row["filename"])
        path_baseline = Path(args.baseline_path) / baseline_filename
        path_target = Path(args.target_path) / row["filename"]
        print(f"Check: target={str(path_target)} baseline={str(path_baseline)}")
        check_tensor_pair(path_baseline=path_baseline, path_target=path_target)
        print()


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

    is_large_diff = max_diff > 1e-3

    print(
        f"{'❌' if is_large_diff else '✅'} "
        f"max_diff={max_diff:.3f} mean_diff={mean_diff:.3f} "
    )

    if is_large_diff:
        print(f"x_baseline(sample)={get_truncated_value(x_baseline)}")
        print(f"x_target(sample)={get_truncated_value(x_target)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str)
    parser.add_argument("--target-path", type=str)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=1000000)
    parser.add_argument("--baseline-start-id", type=int, default=0)
    args = parser.parse_args()
    main(args)
