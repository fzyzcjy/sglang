import argparse
from pathlib import Path


def main(args):
    TODO


def read_meta(path):
    path = Path(path)
    assert path.is_dir()

    return TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str)
    parser.add_argument("--target-path", type=str)
    args = parser.parse_args()
    main(args)
