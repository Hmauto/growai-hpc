#!/bin/bash
#SBATCH --job-name=growai_llm
#SBATCH --time=12:00:00              # 12 hours = 48 GPU hours total
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gpus-per-task=4            # 4x A100 (needed for LLM)
#SBATCH --cpus-per-task=32           # Full node CPUs
#SBATCH --mem=480GB                  # Full node memory
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=aih4a_metafarm     # Your project account

# Grow AI - LLM Fine-tuning (CINECA Optimized)
# Budget: 10,000 GPU hours total | Monthly: ~4,000 GPU hours
# This job: 4 GPUs × 12 hours = 48 GPU hours
# Can run ~208 similar jobs within total budget
# Uses DeepSpeed for efficient training

echo "=========================================="
echo "  Grow AI - LLM Fine-tuning"
echo "  CINECA Leonardo HPC"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Account: aih4a_metafarm"
echo "Budget: 4 GPUs × 12 hours = 48 GPU hours"
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

echo "Starting QLoRA fine-tuning of Qwen 2.5 7B..."
python $HOME/growai/src/train_llm.py \
    --data-path $SCRATCH_DIR/data/llm_training.jsonl \
    --output-dir $RESULTS_DIR \
    --checkpoint-dir $CHECKPOINT_DIR \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --epochs 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --save-every 100

echo "Copying final model to WORK storage..."
cp -r $RESULTS_DIR $WORK_DIR/models/llm/

echo "Completed: $(date)"
echo "Total cost: 48 GPU hours"
echo "Remaining budget: Check with 'saldo -b'"
