from pathlib import Path
from argparse import ArgumentParser


def main(args):
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    TODO


if __name__ == "__main__":
    parser = ArgumentParser(description="Create truncated model for fast debugging.")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    main(parser.parse_args())
