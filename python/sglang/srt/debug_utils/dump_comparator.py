import argparse
from pathlib import Path

import polars as pl


def main(args):
    df_target = read_meta(args.target_path)
    assert all(c in df_target.columns for c in ["rank", "forward_pass_id", "name"])

    for forward_pass_id in sorted(set(df_target["forward_pass_id"].to_list())):
        for rank in sorted(set(df_target["rank"].to_list())):
            names = df_target.filter(
                (pl.col("rank") == rank)
                & (pl.col("forward_pass_id") == forward_pass_id)
            )["name"].to_list()
            for name in names:
                TODO


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str)
    parser.add_argument("--target-path", type=str)
    args = parser.parse_args()
    main(args)
