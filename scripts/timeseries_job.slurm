#!/bin/bash
#SBATCH --job-name=growai_timeseries
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=aih4a_metafarm

# Cost: 4 GPUs × 4 hours = 16 GPU hours

echo "=========================================="
echo "  Grow AI - Time Series Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Cost: 16 GPU hours"
echo "Started: $(date)"
echo "=========================================="

module purge
module load python/3.11.7 profile/deeplrn cuda/11.8
source $HOME/growai/venv/bin/activate

WORK_DIR="$WORK/growai"
SCRATCH_DIR="$SCRATCH/growai"
RESULTS_DIR="${SCRATCH_DIR}/results/timeseries_${SLURM_JOB_ID}"

mkdir -p $RESULTS_DIR
mkdir -p $WORK_DIR/models/timeseries

python $HOME/growai/src/train_timeseries.py \
    --data-path $SCRATCH_DIR/data/sensor_data.csv \
    --output-dir $RESULTS_DIR \
    --max-epochs 50 \
    --batch-size 64

cp -r $RESULTS_DIR $WORK_DIR/models/timeseries/
echo "Completed: $(date)"
