#!/bin/bash
#SBATCH --job-name=real_data_metrics
#SBATCH --output=logs/real_data_metrics_%j.out
#SBATCH --error=logs/real_data_metrics_%j.err
#SBATCH --time=4:00:00          # 4 hours (should be enough for metrics computation)
#SBATCH --nodes=1               # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8       # CPUs for data loading and preprocessing
#SBATCH --gres=gpu:a5000:1      # 1 GPU (metrics computation doesn't need multiple GPUs)
#SBATCH --mem=64gb              # Memory for models and data
#SBATCH --partition=ai4health   # Partition name (adjust to your cluster's partition)
#SBATCH --account=ai4health     # Account name (adjust to your cluster's account)
#SBATCH --mail-type=ALL         # Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mahmoud.ibrahim@vito.be  # Your email (adjust as needed)

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Load modules
module load CUDA
source activate roentgen

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Verify environment
echo "=========================================="
echo "Environment Setup:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to project directory
cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

# Parse arguments
# Usage: sbatch submit_real_data_metrics.sh [config_file] [validation_csv] [num_samples] [output_dir] [batch_size]
CONFIG_FILE="${1:-}"              # Optional config file
VALIDATION_CSV="${2:-}"            # Required: validation data path
NUM_SAMPLES="${3:-}"               # Optional: number of samples (-1 for all)
OUTPUT_DIR="${4:-real_data_metrics}"  # Optional: output directory
BATCH_SIZE="${5:-}"                # Optional: batch size for metrics computation (default: 16)

# Check if validation_csv is provided
if [ -z "$VALIDATION_CSV" ]; then
    echo "ERROR: Validation CSV path is required"
    echo "Usage: sbatch submit_real_data_metrics.sh [config_file] <validation_csv> [num_samples] [output_dir] [batch_size]"
    echo ""
    echo "Examples:"
    echo "  sbatch submit_real_data_metrics.sh configs/test_config.yaml /path/to/validation/data"
    echo "  sbatch submit_real_data_metrics.sh configs/test_config.yaml /path/to/validation/data 200"
    echo "  sbatch submit_real_data_metrics.sh configs/test_config.yaml /path/to/validation/data 200 my_output"
    echo "  sbatch submit_real_data_metrics.sh configs/test_config.yaml /path/to/validation/data 200 my_output 8"
    exit 1
fi

# Check if config file exists (if provided)
if [ -n "$CONFIG_FILE" ] && [ ! -f "$CONFIG_FILE" ]; then
    echo "WARNING: Config file not found: $CONFIG_FILE"
    echo "Continuing without config file..."
    CONFIG_FILE=""
fi

# Build command
CMD="python roentgenv2/train_code/calculate_metrics_on_real_data.py"
CMD="$CMD --validation_csv \"$VALIDATION_CSV\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""

# Add optional arguments
if [ -n "$CONFIG_FILE" ]; then
    CMD="$CMD --config_file \"$CONFIG_FILE\""
fi

if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch_size $BATCH_SIZE"
fi

# Check if validation data is WebDataset format (has .tar files)
if [ -d "$VALIDATION_CSV" ]; then
    TAR_COUNT=$(find "$VALIDATION_CSV" -maxdepth 1 -name "*.tar" 2>/dev/null | wc -l)
    if [ "$TAR_COUNT" -gt 0 ]; then
        echo "Detected WebDataset format (found $TAR_COUNT tar files)"
        CMD="$CMD --use_wds_dataset"
    fi
fi

# Print command
echo "=========================================="
echo "Running real data metrics calculation..."
echo "Validation CSV: $VALIDATION_CSV"
if [ -n "$CONFIG_FILE" ]; then
    echo "Config file: $CONFIG_FILE"
fi
if [ -n "$NUM_SAMPLES" ]; then
    echo "Number of samples: $NUM_SAMPLES"
else
    echo "Number of samples: ALL"
fi
echo "Output directory: $OUTPUT_DIR"
if [ -n "$BATCH_SIZE" ]; then
    echo "Batch size: $BATCH_SIZE"
else
    echo "Batch size: 16 (default)"
fi
echo "=========================================="
echo "Command: $CMD"
echo "=========================================="

# Run the script
eval $CMD

# Capture exit code
EXIT_CODE=$?

echo "=========================================="
echo "Real data metrics calculation finished"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# Exit with the same code as the script
exit $EXIT_CODE

