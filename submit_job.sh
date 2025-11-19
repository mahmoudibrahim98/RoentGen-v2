#!/bin/bash
# Helper script to ensure logs directory exists before submitting SLURM job

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if logs directory was created successfully
if [ ! -d "logs" ]; then
    echo "ERROR: Failed to create logs directory"
    exit 1
fi

echo "✓ Logs directory ready"
echo "Submitting SLURM job..."

# Submit the job
sbatch submit_training.sh "$@"

# Get the job ID
JOB_ID=$(squeue -u $USER --format="%i" --noheader --name=roentgen_training | head -1)

if [ -n "$JOB_ID" ]; then
    echo "✓ Job submitted: $JOB_ID"
    echo "Monitor with: squeue -j $JOB_ID"
    echo "View logs with: tail -f logs/slurm_${JOB_ID}.out"
else
    echo "Job submitted (check status with: squeue -u \$USER)"
fi

