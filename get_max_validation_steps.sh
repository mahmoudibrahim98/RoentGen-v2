#!/bin/bash
# Script to find the maximum validation step for each trained model
# Usage: ./get_max_validation_steps.sh [--csv|--json]

set -e

# Parse arguments
OUTPUT_FORMAT="table"
if [ "$1" == "--csv" ]; then
    OUTPUT_FORMAT="csv"
elif [ "$1" == "--json" ]; then
    OUTPUT_FORMAT="json"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if jq is available (for JSON parsing)
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}Warning: jq not found. Using Python for JSON parsing instead.${NC}"
    USE_PYTHON=true
else
    USE_PYTHON=false
fi

# Function to extract max step from manifest using jq
get_max_step_jq() {
    local manifest_file="$1"
    local max_step=$(jq -r 'keys[] | select(startswith("checkpoint-")) | sub("checkpoint-"; "") | tonumber' "$manifest_file" 2>/dev/null | sort -n | tail -1)
    echo "$max_step"
}

# Function to extract max step from manifest using Python
get_max_step_python() {
    local manifest_file="$1"
    python3 << EOF
import json
import sys

try:
    with open("$manifest_file", 'r') as f:
        data = json.load(f)
    
    max_step = 0
    for key in data.keys():
        if key.startswith("checkpoint-"):
            try:
                step = int(key.split("-")[1])
                max_step = max(max_step, step)
            except (ValueError, IndexError):
                pass
    
    print(max_step if max_step > 0 else "")
except Exception as e:
    print("")
    sys.exit(1)
EOF
}

# Get the base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUTS_DIR="${BASE_DIR}/outputs"

if [ ! -d "$OUTPUTS_DIR" ]; then
    echo -e "${RED}Error: outputs directory not found at $OUTPUTS_DIR${NC}"
    exit 1
fi

echo "Scanning outputs directory for validation manifests..."
echo "=================================================="
echo ""

# Array to store results
declare -a results

# Loop through all output directories (output, output_v2, output_v3, etc.)
for output_dir in "$OUTPUTS_DIR"/*/; do
    if [ ! -d "$output_dir" ]; then
        continue
    fi
    
    output_name=$(basename "$output_dir")
    
    # Loop through each experiment subdirectory
    for exp_dir in "$output_dir"*/; do
        if [ ! -d "$exp_dir" ]; then
            continue
        fi
        
        exp_name=$(basename "$exp_dir")
        manifest_file="${exp_dir}validation_manifest.json"
        
        # Check if validation manifest exists
        if [ ! -f "$manifest_file" ]; then
            continue
        fi
        
        # Extract experiment path (relative to outputs)
        exp_path="outputs/${output_name}/${exp_name}"
        
        # Get max step
        if [ "$USE_PYTHON" = true ]; then
            max_step=$(get_max_step_python "$manifest_file")
        else
            max_step=$(get_max_step_jq "$manifest_file")
        fi
        
        # Only add if we found a valid step
        if [ -n "$max_step" ] && [ "$max_step" != "0" ] && [ "$max_step" != "" ]; then
            results+=("${exp_path}|${max_step}")
        fi
    done
done

# Sort results by max step (descending)
IFS=$'\n' sorted_results=($(printf '%s\n' "${results[@]}" | sort -t'|' -k2 -rn))
unset IFS

# Output based on format
if [ "$OUTPUT_FORMAT" == "csv" ]; then
    echo "experiment,max_step"
    for result in "${sorted_results[@]}"; do
        exp_path=$(echo "$result" | cut -d'|' -f1)
        max_step=$(echo "$result" | cut -d'|' -f2)
        echo "$exp_path,$max_step"
    done
elif [ "$OUTPUT_FORMAT" == "json" ]; then
    echo "["
    first=true
    for result in "${sorted_results[@]}"; do
        exp_path=$(echo "$result" | cut -d'|' -f1)
        max_step=$(echo "$result" | cut -d'|' -f2)
        if [ "$first" = true ]; then
            first=false
        else
            echo ","
        fi
        echo -n "  {\"experiment\": \"$exp_path\", \"max_step\": $max_step}"
    done
    echo ""
    echo "]"
else
    # Print header
    echo -e "${GREEN}Experiment${NC} | ${GREEN}Max Validation Step${NC}"
    echo "----------------------------------------"
    
    # Print results
    for result in "${sorted_results[@]}"; do
        exp_path=$(echo "$result" | cut -d'|' -f1)
        max_step=$(echo "$result" | cut -d'|' -f2)
        printf "%-50s | %s\n" "$exp_path" "$max_step"
    done
    
    echo ""
    echo "=================================================="
    echo "Total experiments with validation: ${#sorted_results[@]}"
fi

