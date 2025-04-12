#!/bin/bash

# Check if filename is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <jsonl_file>"
    exit 1
fi

JSONL_FILE=$1

# Check if file exists
if [ ! -f "$JSONL_FILE" ]; then
    echo "Error: File not found: $JSONL_FILE"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install it first."
    exit 1
fi

# Extract the max and min dates
echo "Extracting dates from $JSONL_FILE..."
MAX_DATE=$(jq -s 'max_by(.generation_datetime) | .generation_datetime' "$JSONL_FILE" | sed 's/"//g')
MIN_DATE=$(jq -s 'min_by(.generation_datetime) | .generation_datetime' "$JSONL_FILE" | sed 's/"//g')

# Convert dates to timestamps
MAX_TS=$(date -d "$MAX_DATE" +%s)
MIN_TS=$(date -d "$MIN_DATE" +%s)

# Calculate difference
DIFF_SECONDS=$((MAX_TS - MIN_TS))
DIFF_MINUTES=$((DIFF_SECONDS / 60)) 
DIFF_HOURS=$((DIFF_SECONDS / 3600))
DIFF_DAYS=$((DIFF_SECONDS / 86400))

# Display results
echo "==================== RESULTS ===================="
echo "Earliest datetime: $MIN_DATE"
echo "Latest datetime:   $MAX_DATE"
echo "-------------------------------------------"
echo "Difference: $DIFF_SECONDS seconds"
echo "           $DIFF_MINUTES minutes"
echo "           $DIFF_HOURS hours"
echo "           $DIFF_DAYS days"
echo "==============================================="

# Exit successfully
exit 0 