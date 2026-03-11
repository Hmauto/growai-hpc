#!/bin/bash
#SBATCH --job-name=growai_recommendation
#SBATCH --time=02:00:00              # 2 hours = 8 GPU hours total
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gpus-per-task=2            # 2x A100 (sufficient for this task)
#SBATCH --cpus-per-task=16           # Half node CPUs
#SBATCH --mem=240GB                  # Half node memory
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=aih4a_metafarm     # Your project account

# Grow AI - Recommendation Training (CINECA Optimized)
# Budget: 10,000 GPU hours total | Monthly: ~4,000 GPU hours
# This job: 2 GPUs × 2 hours = 4 GPU hours
# Can run ~2,500 similar jobs within total budget

echo "=========================================="
echo "  Grow AI - Recommendation Training"
echo "  CINECA Leonardo HPC"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Account: aih4a_metafarm"
echo "Budget: 2 GPUs × 2 hours = 4 GPU hours"
echo "Started: $(date)"
echo "=========================================="

module purge
module load python/3.11.7 profile/deeplrn cuda/11.8
source $HOME/growai/venv/bin/activate

WORK_DIR="$WORK/growai"
SCRATCH_DIR="$SCRATCH/growai"
RESULTS_DIR="${SCRATCH_DIR}/results/recommendation_${SLURM_JOB_ID}"

mkdir -p $RESULTS_DIR
mkdir -p $WORK_DIR/models/recommendation

echo "Training XGBoost + Neural Network models..."
python $HOME/growai/src/train_recommendation.py \
    --data-path $SCRATCH_DIR/data/training_data.csv \
    --output-dir $RESULTS_DIR \
    --model both

echo "Copying results to WORK storage..."
cp -r $RESULTS_DIR $WORK_DIR/models/recommendation/

echo "Completed: $(date)"
echo "Total cost: 4 GPU hours"
