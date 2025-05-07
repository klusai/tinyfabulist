import argparse

from tinyfabulist.evaluate.en import add_evaluate_subparser
from tinyfabulist.evaluate.ro import add_evaluate_ro_subparser
from tinyfabulist.generate import add_generate_subparser, add_translate_subparser
from tinyfabulist.stats import add_stats_subparser
from tinyfabulist.visualizer.jsonl_visualizer import add_visualize_subparser


def main():
    parser = argparse.ArgumentParser(
        description="TinyFabulist - Fable generator and evaluator"
    )
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    
    # Generates prompts or complete fables using selected models
    add_generate_subparser(subparsers)
    
    # Evaluates generated fables for quality and adherence to prompts
    # - English evaluation for original fables
    # - Romanian evaluation for translated fables
    add_evaluate_subparser(subparsers)
    add_evaluate_ro_subparser(subparsers)

    # Computes and displays aggregated statistics from evaluation results
    add_stats_subparser(subparsers)

    # Translates fables to other languages using various translation engines
    add_translate_subparser(subparsers)

    # Add visualization dashboard functionality
    add_visualize_subparser(subparsers)

    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()