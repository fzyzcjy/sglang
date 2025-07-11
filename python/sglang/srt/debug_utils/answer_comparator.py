import polars as pl
import argparse


def main(args):
    df = pl.concat([
        _read_df_raw(p, category=category, trial_index=i)
        for category, paths in [
            ("baseline", args.baseline_path),
            ("target", args.target_path),
        ]
        for i, p in paths
    ])

    TODO


def _read_df_raw(path: str, category: str, trial_index: int):
    return pl.read_json(path).with_columns(category=pl.lit(category), trial_index=trial_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str, nargs="+")
    parser.add_argument("--target-path", type=str, nargs="+")
    args = parser.parse_args()
    main(args)
