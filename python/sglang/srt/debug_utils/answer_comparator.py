import argparse

def main(args):
    TODO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, nargs="+")
    parser.add_argument("--target", type=str, nargs="+")
    args = parser.parse_args()
    main(args)
