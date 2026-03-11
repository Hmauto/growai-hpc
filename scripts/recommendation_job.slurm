#!/bin/bash
#SBATCH --job-name=growai_recommendation
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=aih4a_metafarm

# Cost: 2 GPUs × 2 hours = 4 GPU hours

echo "=========================================="
echo "  Grow AI - Recommendation Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Cost: 4 GPU hours"
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

python $HOME/growai/src/train_recommendation.py \
    --data-path $SCRATCH_DIR/data/training_data.csv \
    --output-dir $RESULTS_DIR \
    --model both

cp -r $RESULTS_DIR $WORK_DIR/models/recommendation/
echo "Completed: $(date)"
