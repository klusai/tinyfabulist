#!/bin/bash
set -e  # Exit immediately if any command exits with a non-zero status

# Determine the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}"

# Check if the AI model parameter is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <ai_model>"
    exit 1
fi

MODEL="$1"

# Define absolute output file paths based on the provided model name
ARTIFACTS_DIR="${SCRIPT_DIR}/src/artifacts"
GENERATION_OUTPUT="${ARTIFACTS_DIR}/${MODEL}.csv"
EVALUATION_OUTPUT="${ARTIFACTS_DIR}/${MODEL}-evaluation_results.json"
DIVERSITY_OUTPUT="${ARTIFACTS_DIR}/${MODEL}-diversity_evaluation.json"

# Clear existing files if they exist
[ -f "${EVALUATION_OUTPUT}" ] && rm "${EVALUATION_OUTPUT}"
[ -f "${DIVERSITY_OUTPUT}" ] && rm "${DIVERSITY_OUTPUT}"

# Create a temporary YAML file for evaluation configuration
TEMP_YAML=$(mktemp)
cp "${SCRIPT_DIR}/src/evaluation/config.yml" "$TEMP_YAML"

echo "Running fable evaluations for model: ${MODEL}..."
python "${SCRIPT_DIR}/src/cli.py" evaluate \
    --csv_input "${GENERATION_OUTPUT}" \
    --yaml_input "${TEMP_YAML}" \
    --evaluation_output "${EVALUATION_OUTPUT}" \
    --diversity_number 25 \
    --diversity_output "${DIVERSITY_OUTPUT}"

# Clean up the temporary YAML file
rm "$TEMP_YAML"
