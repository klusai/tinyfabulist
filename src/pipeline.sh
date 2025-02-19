#!/bin/bash
set -e  # Exit immediately if any command exits with a non-zero status

# Check if the AI model parameter is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <ai_model>"
    exit 1
fi

MODEL="$1"

# Define output file paths based on the provided model name
# Clear existing files if they exist
if [ -f "src/artifacts/${MODEL}.csv" ]; then
    rm "src/artifacts/${MODEL}.csv"
fi
if [ -f "src/artifacts/${MODEL}-evaluation_results.json" ]; then
    rm "src/artifacts/${MODEL}-evaluation_results.json"
fi
if [ -f "src/artifacts/${MODEL}-diversity_evaluation.json" ]; then
    rm "src/artifacts/${MODEL}-diversity_evaluation.json"
fi

GENERATION_OUTPUT="src/artifacts/${MODEL}.csv"
EVALUATION_OUTPUT="src/artifacts/${MODEL}-evaluation_results.json"
DIVERSITY_OUTPUT="src/artifacts/${MODEL}-diversity_evaluation.json"

echo "Running fable generation for model: ${MODEL}..."
python -m src.cli generate \
    --model "${MODEL}" \
    --config_path "src/generation/config.yml" \
    --output_file "${GENERATION_OUTPUT}" \

# Create a temporary YAML file
TEMP_YAML=$(mktemp)
cp src/evaluation/config.yml "$TEMP_YAML"

echo "Running fable evaluations for model: ${MODEL}..."
python -m src.cli evaluate \
    --csv_input "${GENERATION_OUTPUT}" \
    --yaml_input "${TEMP_YAML}" \
    --evaluation_output "${EVALUATION_OUTPUT}" \
    --diversity_number 25 \
    --diversity_output "${DIVERSITY_OUTPUT}"

# Clean up the temporary YAML file
rm "$TEMP_YAML"
