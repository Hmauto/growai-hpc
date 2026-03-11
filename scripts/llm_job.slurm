#!/bin/bash
#SBATCH --job-name=growai_llm
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=aih4a_metafarm

# Cost: 4 GPUs × 12 hours = 48 GPU hours

echo "=========================================="
echo "  Grow AI - LLM Fine-tuning"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Cost: 48 GPU hours"
echo "Started: $(date)"
echo "=========================================="

module purge
module load python/3.11.7 profile/deeplrn cuda/11.8
source $HOME/growai/venv/bin/activate

WORK_DIR="$WORK/growai"
SCRATCH_DIR="$SCRATCH/growai"
RESULTS_DIR="${SCRATCH_DIR}/results/llm_${SLURM_JOB_ID}"
CHECKPOINT_DIR="${WORK_DIR}/checkpoints/llm_${SLURM_JOB_ID}"

mkdir -p $RESULTS_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p $WORK_DIR/models/llm

python $HOME/growai/src/train_llm.py \
    --data-path $SCRATCH_DIR/data/llm_training.jsonl \
    --output-dir $RESULTS_DIR \
    --checkpoint-dir $CHECKPOINT_DIR \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --epochs 3 \
    --batch-size 4

cp -r $RESULTS_DIR $WORK_DIR/models/llm/
echo "Completed: $(date)"
