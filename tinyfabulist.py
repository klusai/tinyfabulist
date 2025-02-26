import argparse
from comparate import add_comparator_subparser
from generate import add_generate_subparser
from evaluate import add_evaluate_subparser

def main():
    parser = argparse.ArgumentParser(description='TinyFabulist - Fable generator and evaluator')
    subparsers = parser.add_subparsers(title='Commands', dest='command')
    add_generate_subparser(subparsers)
    add_evaluate_subparser(subparsers)
    add_comparator_subparser(subparsers)
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
