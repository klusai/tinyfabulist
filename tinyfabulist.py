import argparse

from tinyfabulist.evaluate.en import add_evaluate_subparser
from tinyfabulist.generate import add_generate_subparser
from tinyfabulist.stats import add_stats_subparser
from tinyfabulist.translate.controller import get_translate_subparser
from tinyfabulist.translate.controller import add_translate_subparser




def main():
    parser = argparse.ArgumentParser(
        description="TinyFabulist - Fable generator and evaluator"
    )
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    add_generate_subparser(subparsers)
    add_evaluate_subparser(subparsers)
    add_stats_subparser(subparsers)
    add_translate_subparser(subparsers)  # Add our translation subparser with engine selection
    
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()