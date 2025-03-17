import argparse

from evaluate.en import add_evaluate_subparser
from generate import add_generate_subparser
from stats import add_stats_subparser
from translate.deepl import deepl_subparser


def main():
    parser = argparse.ArgumentParser(
        description="TinyFabulist - Fable generator and evaluator"
    )
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    add_generate_subparser(subparsers)
    add_evaluate_subparser(subparsers)
    add_stats_subparser(subparsers)
    deepl_subparser(subparsers)
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
