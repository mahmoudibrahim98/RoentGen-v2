#!/bin/bash
# Script to create training/validation submit scripts and log directories
# Usage: ./create_submit_script.sh [train|validate] <experiment_name>
# Example: ./create_submit_script.sh train v4/3_demographic_encoder_v4_with_dropout

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 [train|validate] <experiment_name>"
    echo "Example: $0 train v4/3_demographic_encoder_v4_with_dropout"
    echo "Example: $0 validate v4/3_demographic_encoder_v4_with_dropout"
    exit 1
fi

MODE=$1
EXPERIMENT_NAME=$2

# Validate mode
if [ "$MODE" != "train" ] && [ "$MODE" != "validate" ]; then
    echo "Error: Mode must be 'train' or 'validate'"
    exit 1
fi

# Parse experiment name (e.g., v4/3_demographic_encoder_v4_with_dropout or v1.5/1_hcn_with_aux_loss_and_dropout)
if [[ "$EXPERIMENT_NAME" =~ ^(v[0-9]+(\.[0-9]+)?)/(.+)$ ]]; then
    VERSION="${BASH_REMATCH[1]}"
    EXPERIMENT_ID="${BASH_REMATCH[3]}"
else
    echo "Error: Experiment name must be in format 'vX/experiment_name' or 'vX.Y/experiment_name'"
    echo "Example: v4/3_demographic_encoder_v4_with_dropout"
    echo "Example: v1.5/1_hcn_with_aux_loss_and_dropout"
    exit 1
fi

# Extract number prefix (e.g., "3" from "3_demographic_encoder_v4_with_dropout")
if [[ "$EXPERIMENT_ID" =~ ^([0-9]+)_(.+)$ ]]; then
    EXP_NUMBER="${BASH_REMATCH[1]}"
    EXP_NAME_WITHOUT_NUMBER="${BASH_REMATCH[2]}"
else
    echo "Warning: Could not extract experiment number from '$EXPERIMENT_ID'"
    EXP_NUMBER=""
    EXP_NAME_WITHOUT_NUMBER="$EXPERIMENT_ID"
fi

# Extract short name for script (remove _v4 suffix if present and number prefix)
# Remove patterns like _v4_ or _v4$ (version suffix)
SHORT_NAME=$(echo "$EXP_NAME_WITHOUT_NUMBER" | sed 's/_v[0-9]\+_/_/g' | sed 's/_v[0-9]\+$//')

# Create log directory
LOG_DIR="logs/logs_${VERSION}/${EXPERIMENT_ID}"
echo "Creating log directory: $LOG_DIR"
mkdir -p "$LOG_DIR"

# Create submit_jobs directory if it doesn't exist
SUBMIT_DIR="submit_jobs/${VERSION}"
mkdir -p "$SUBMIT_DIR"

# Determine script name
# Remove "train_" prefix from SHORT_NAME if present (we'll add it back in the script name)
SCRIPT_BASE_NAME=$(echo "$SHORT_NAME" | sed 's/^train_//')
if [ "$MODE" == "train" ]; then
    SCRIPT_NAME="${EXP_NUMBER}_submit_train_${SCRIPT_BASE_NAME}.sh"
else
    SCRIPT_NAME="${EXP_NUMBER}_submit_validation_${SCRIPT_BASE_NAME}.sh"
fi

SCRIPT_PATH="${SUBMIT_DIR}/${SCRIPT_NAME}"

echo "Creating submit script: $SCRIPT_PATH"

# Generate training script
if [ "$MODE" == "train" ]; then
    cat > "$SCRIPT_PATH" << 'TRAIN_EOF'
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/logs_VERSION/EXPERIMENT_ID/training_%j.out
#SBATCH --error=logs/logs_VERSION/EXPERIMENT_ID/training_%j.err
#SBATCH --time=7-00:00:00          # 7 days (adjust as needed)
#SBATCH --nodes=1                  # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         # CPUs per task (adjust based on your needs)
#SBATCH --gres=gpu:6               # 6 GPUs (matching your training setup)
#SBATCH --mem=90gb                    # Use all available memory on the node
#SBATCH --partition=precisionhealth      # Partition name (adjust to your cluster's partition)
#SBATCH --account=precisionhealth      # Account name (adjust to your cluster's account)
#SBATCH --mail-type=ALL             # Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mahmoud.ibrahim@vito.be  # Your email (adjust as needed)

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Load modules if needed (uncomment and adjust for your cluster)
# module load cuda/11.8
# module load python/3.10

module load CUDA
source activate roentgen

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Set environment variables for distributed training
# For single-node multi-GPU, we don't need InfiniBand - use local GPU interconnects
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,
export NCCL_DEBUG=WARN  # Set to INFO for debugging, WARN for less verbose
export NCCL_IB_DISABLE=1  # Disable InfiniBand (not needed for single-node)
export NCCL_P2P_DISABLE=0  # Enable P2P (NVLink/PCIe) for single-node
export NCCL_SHM_DISABLE=0  # Enable shared memory
# Don't set NCCL_SOCKET_IFNAME for single-node - let NCCL auto-detect
# export NCCL_SOCKET_IFNAME=ib0  # Only needed for multi-node with InfiniBand


# Navigate to project directory
cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

# Create logs directory if it doesn't exist
mkdir -p logs/logs_VERSION/EXPERIMENT_ID

# Config file path (adjust as needed)
CONFIG_FILE="configs/VERSION/EXPERIMENT_ID.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Launch training with accelerate
echo "=========================================="
echo "Starting training..."
echo "Config file: $CONFIG_FILE"
echo "=========================================="

accelerate launch \
    --num_processes=6 \
    --multi_gpu \
    --mixed_precision bf16 \
    roentgenv2/train_code/train.py \
    --config_file="$CONFIG_FILE"

# Capture exit code
EXIT_CODE=$?

echo "=========================================="
echo "Training finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# Exit with the same code as the training script
exit $EXIT_CODE
TRAIN_EOF

    # Construct config file name (add train_ prefix if not present)
    if [[ "$EXPERIMENT_ID" =~ train_ ]]; then
        CONFIG_NAME="$EXPERIMENT_ID"
    else
        # Add train_ after the number prefix (e.g., 3_demographic_encoder_v4_with_dropout -> 3_train_demographic_encoder_v4_with_dropout)
        CONFIG_NAME=$(echo "$EXPERIMENT_ID" | sed 's/^\([0-9]*\)_\(.*\)/\1_train_\2/')
    fi
    
    # Replace placeholders in training script
    sed -i "s|logs/logs_VERSION/EXPERIMENT_ID|logs/logs_${VERSION}/${EXPERIMENT_ID}|g" "$SCRIPT_PATH"
    sed -i "s|configs/VERSION/EXPERIMENT_ID.yaml|configs/${VERSION}/${CONFIG_NAME}.yaml|g" "$SCRIPT_PATH"
    sed -i "s|mkdir -p logs/logs_VERSION/EXPERIMENT_ID|mkdir -p logs/logs_${VERSION}/${EXPERIMENT_ID}|g" "$SCRIPT_PATH"

else
    # Generate validation script
    cat > "$SCRIPT_PATH" << 'VAL_EOF'
#!/bin/bash
#SBATCH --job-name=validation_monitor
#SBATCH --output=logs/logs_VERSION/EXPERIMENT_ID/validation_%j.out
#SBATCH --error=logs/logs_VERSION/EXPERIMENT_ID/validation_%j.err
#SBATCH --time=7-00:00:00          # 2 days (validation can run longer)
#SBATCH --nodes=1                  # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # CPUs per task
#SBATCH --gres=gpu:rtx4090:8               # 8 GPUs for distributed validation
#SBATCH --mem=256gb                # Memory for validation (8 processes × models + metrics)
#SBATCH --partition=ai4health      # Partition name (adjust to your cluster's partition)
#SBATCH --account=ai4health      # Account name (adjust to your cluster's account)
#SBATCH --mail-type=ALL             # Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mahmoud.ibrahim@vito.be  # Your email (adjust as needed)

# Usage:
#   sbatch submit_validation.sh [config_file] [check_interval] [manifest_file] [load_images_flag]
#
# Arguments:
#   config_file (optional): Path to config file (default: configs/test_config.yaml)
#   check_interval (optional): Check for new checkpoints every N seconds (default: 1800)
#   manifest_file (optional): Manifest file name (default: validation_manifest.json)
#   load_images_flag (optional): Set to "1" or "true" to load pre-generated images from validation_images directory
#
# Examples:
#   # Normal mode: Monitor checkpoints and generate images
#   sbatch submit_validation.sh configs/my_config.yaml
#
#   # Load pre-generated images from validation_images directory
#   sbatch submit_validation.sh configs/my_config.yaml 1800 validation_manifest.json 1

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

# Set environment variables for multi-GPU distributed validation
# SLURM automatically sets CUDA_VISIBLE_DEVICES when --gres=gpu is used
# Don't override it - let SLURM handle GPU allocation
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1  # Disable InfiniBand (not needed for single-node)
export NCCL_P2P_DISABLE=0  # Enable P2P (NVLink/PCIe) for single-node
export NCCL_SHM_DISABLE=0  # Enable shared memory

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to project directory
cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

# Config file path (adjust as needed)
CONFIG_FILE="${1:-configs/VERSION/EXPERIMENT_ID.yaml}"  # Use first argument or default

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available config files:"
    ls -1 configs/*.yaml 2>/dev/null || echo "  No config files found in configs/"
    exit 1
fi

# Optional: Check interval, manifest file, load images flag, and stop-at-step arguments
CHECK_INTERVAL="${2:-300}"  # Default: 30 minutes
MANIFEST_FILE="${3:-validation_manifest.json}"
LOAD_IMAGES_FLAG="${4:-0}"  # Optional: set to "1" or "true" to load pre-generated images
STOP_AT_STEP="${5:-}"      # Optional: if set, stop validation after this global step

# Launch validation monitor
echo "=========================================="
echo "Starting validation monitoring..."
echo "Config file: $CONFIG_FILE"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "Manifest file: $MANIFEST_FILE"
if [ "$LOAD_IMAGES_FLAG" = "1" ] || [ "$LOAD_IMAGES_FLAG" = "true" ]; then
    echo "Mode: Loading pre-generated images from validation_images directory (skipping generation)"
else
    echo "Mode: Generating images from checkpoints"
fi
echo "=========================================="

# Build accelerate launch command
ACCELERATE_CMD="accelerate launch \
    --num_processes=8 \
    --num_machines=1 \
    --multi_gpu \
    --mixed_precision bf16 \
    roentgenv2/train_code/run_validation_monitor_debug.py \
    --config_file=\"$CONFIG_FILE\" \
    --check_interval=\"$CHECK_INTERVAL\" \
    --manifest_file=\"$MANIFEST_FILE\""

# Add stop_at_step if provided
if [ -n "$STOP_AT_STEP" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD --stop_at_step=\"$STOP_AT_STEP\""
fi

# Add load_images_from_dir flag if provided
if [ "$LOAD_IMAGES_FLAG" = "1" ] || [ "$LOAD_IMAGES_FLAG" = "true" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD --load_images_from_dir"
fi

# Run validation with all allocated GPUs
# Using 8 GPUs for faster validation (distributes image generation across GPUs)
# Note: When loading from directory, GPUs are only used for metrics computation
eval $ACCELERATE_CMD

# Capture exit code
EXIT_CODE=$?

echo "=========================================="
echo "Validation monitoring finished"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# Exit with the same code as the validation script
exit $EXIT_CODE
VAL_EOF

    # Construct config file name (add train_ prefix if not present)
    if [[ "$EXPERIMENT_ID" =~ train_ ]]; then
        CONFIG_NAME="$EXPERIMENT_ID"
    else
        # Add train_ after the number prefix (e.g., 3_demographic_encoder_v4_with_dropout -> 3_train_demographic_encoder_v4_with_dropout)
        CONFIG_NAME=$(echo "$EXPERIMENT_ID" | sed 's/^\([0-9]*\)_\(.*\)/\1_train_\2/')
    fi
    
    # Replace placeholders in validation script
    sed -i "s|logs/logs_VERSION/EXPERIMENT_ID|logs/logs_${VERSION}/${EXPERIMENT_ID}|g" "$SCRIPT_PATH"
    sed -i "s|configs/VERSION/EXPERIMENT_ID.yaml|configs/${VERSION}/${CONFIG_NAME}.yaml|g" "$SCRIPT_PATH"
fi

# Make script executable
chmod +x "$SCRIPT_PATH"

echo "=========================================="
echo "Successfully created:"
echo "  Log directory: $LOG_DIR"
echo "  Submit script: $SCRIPT_PATH"
echo "=========================================="

