#!/bin/bash
set -e  # Exit immediately if any command exits with a non-zero status

# Example 1: Generate fables.
echo "Running fable generation..."
python src/cli.py generate \
    --config_path "src/generation/config.yml" \
    --output_file "src/artifacts/fables_with_meta.csv" \
    --num_fables 10

# Example 2: Evaluate generated fables.
echo "Running fable evaluations..."
python src/cli.py evaluate \
    --csv_input "src/artifacts/fables_with_meta.csv" \
    --yaml_input "src/evaluation/config.yml" \
    --evaluation_output "src/artifacts/evaluation_results.json" \
    --diversity_number 20 \
    --diversity_output "src/artifacts/diversity_evaluation.json"

# Example 3: Plot evaluation results.
echo "Plotting results..."
python src/cli.py plot

