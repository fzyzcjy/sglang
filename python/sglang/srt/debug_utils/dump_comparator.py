import polars as pl
import argparse
from pathlib import Path


def main(args):
    TODO


def read_meta(path):
    path = Path(path)
    assert path.is_dir()

    rows = []
    for p in path.glob("*.pt"):
        full_kwargs = {}
        for kv in p.stem.split("___"):
            k, v = kv.split("=")
            full_kwargs[k] = v
        rows.append({
            path: str(p),
            **full_kwargs,
        })

    return pl.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str)
    parser.add_argument("--target-path", type=str)
    args = parser.parse_args()
    main(args)
