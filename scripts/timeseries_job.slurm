#!/bin/bash
#SBATCH --job-name=growai_timeseries
#SBATCH --time=04:00:00              # 4 hours = 16 GPU hours total
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gpus-per-task=4            # 4x A100
#SBATCH --cpus-per-task=32           # Full node CPUs
#SBATCH --mem=480GB                  # Efficient memory
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=aih4a_metafarm     # Your project account

# Grow AI - Time Series Training (CINECA Optimized)
# Budget: 10,000 GPU hours total | Monthly: ~4,000 GPU hours
# This job: 4 GPUs × 4 hours = 16 GPU hours
# Can run ~625 similar jobs within total budget

echo "=========================================="
echo "  Grow AI - Time Series Training"
echo "  CINECA Leonardo HPC"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Account: aih4a_metafarm"
echo "Budget: 4 GPUs × 4 hours = 16 GPU hours"
echo "Started: $(date)"
echo "=========================================="

# Load CINECA modules
module purge
module load python/3.11.7
module load profile/deeplrn
module load cuda/11.8

echo ""
echo "Environment:"
source $HOME/growai/venv/bin/activate
python --version
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Paths (SCRATCH for temp, WORK for persistent)
GROWAI_DIR="$HOME/growai"
WORK_DIR="$WORK/growai"
SCRATCH_DIR="$SCRATCH/growai"
RESULTS_DIR="${SCRATCH_DIR}/results/timeseries_${SLURM_JOB_ID}"

mkdir -p $RESULTS_DIR
mkdir -p $WORK_DIR/models/timeseries

echo ""
echo "Starting Temporal Fusion Transformer training..."
python $GROWAI_DIR/src/train_timeseries.py \
    --data-path $SCRATCH_DIR/data/sensor_data.csv \
    --output-dir $RESULTS_DIR \
    --max-epochs 50 \
    --batch-size 64 \
    --hidden-size 160

echo ""
echo "Copying results to WORK storage..."
cp -r $RESULTS_DIR $WORK_DIR/models/timeseries/

echo "Completed: $(date)"
echo "Total cost: 16 GPU hours"
