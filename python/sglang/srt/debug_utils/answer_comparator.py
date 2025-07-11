import argparse
from typing import List


def main(args):
    baseline_paths: List[str] = args.baseline_path
    target_paths: List[str] = args.target_path

    TODO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str, nargs="+")
    parser.add_argument("--target-path", type=str, nargs="+")
    args = parser.parse_args()
    main(args)
