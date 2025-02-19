import sys
import os

from src.benchmark.plotter import plot_from_artifacts

current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from src.evaluation.core import execute_evaluations
from src.generation.fable_generator import FableGenerator

def main():
    parser = argparse.ArgumentParser(description="Fable Generation, Evaluation, and Plotting CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands: generate, evaluate, plot")

    # Subparser for the 'generate' command.
    parser_generate = subparsers.add_parser("generate", help="Generate fables")
    parser_generate.add_argument("--model", type=str, required=False, help="The AI Model used.")
    parser_generate.add_argument("--config_path", type=str, required=False, help="Path to the generator configuration file")
    parser_generate.add_argument("--output_file", type=str, required=False, help="Output file for generated fables")
    parser_generate.add_argument("--num_fables", type=int, required=False, help="Number of fables to generate")

    # Subparser for the 'evaluate' command.
    parser_evaluate = subparsers.add_parser("evaluate", help="Evaluate generated fables")
    parser_evaluate.add_argument("--csv_input", type=str, required=True, help="Path to CSV input file with fables and metadata")
    parser_evaluate.add_argument("--yaml_input", type=str, required=True, help="Path to YAML configuration file")
    parser_evaluate.add_argument("--evaluation_output", type=str, required=False, help="Output file for individual evaluation results")
    parser_evaluate.add_argument("--diversity_number", type=int, required=False, help="Number of fables for diversity evaluation")
    parser_evaluate.add_argument("--diversity_output", type=str, required=False, help="Output file for diversity evaluation results")

    # Subparser for the 'plot' command.
    parser_plot = subparsers.add_parser("plot", help="Plot evaluation results")

    args = parser.parse_args()

    if args.command == "generate":
        fable_generator = FableGenerator(model=args.model, config_path=args.config_path, output_file=args.output_file, num_fables=100)
        fable_generator.run()
    elif args.command == "evaluate":
        execute_evaluations(args.csv_input, args.yaml_input, args.diversity_number, args.evaluation_output, args.diversity_output)
    elif args.command == "plot":
            artifacts_directory = 'src/artifacts' 
            plot_from_artifacts(artifacts_directory)

if __name__ == "__main__":
    main()
