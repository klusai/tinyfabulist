#!/bin/bash
set -e  # Exit immediately if any command exits with a non-zero status

# Check if the AI model parameter is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <ai_model>"
    exit 1
fi

MODEL="$1"

# Define output file paths based on the provided model name
GENERATION_OUTPUT="src/artifacts/${MODEL}.csv"
EVALUATION_OUTPUT="src/artifacts/${MODEL}-evaluation_results.json"
DIVERSITY_OUTPUT="src/artifacts/${MODEL}-diversity_evaluation.json"

echo "Running fable generation for model: ${MODEL}..."
python src/cli.py generate \
    --model "${MODEL}" \
    --config_path "src/generation/config.yml" \
    --output_file "${GENERATION_OUTPUT}"

echo "Running fable evaluations for model: ${MODEL}..."
python src/cli.py evaluate \
    --csv_input "${GENERATION_OUTPUT}" \
    --yaml_input "src/evaluation/config.yml" \
    --evaluation_output "${EVALUATION_OUTPUT}" \
    --diversity_number 25 \
    --diversity_output "${DIVERSITY_OUTPUT}"
