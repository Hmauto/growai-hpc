#!/bin/bash
# =============================================================================
# GrowAI HPC Master Launcher - CINECA Leonardo Optimized
# =============================================================================
# Budget: 10,000 GPU hours | Monthly: ~4,000 GPU hours
# Project: aih4a_metafarm | Ends: March 31, 2026
#
# Usage: bash run_all.sh [options]
# =============================================================================

set -e

GROWAI_DIR="$HOME/growai"
WORK_DIR="$WORK/growai"
LOG_FILE="$WORK_DIR/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"

# Create necessary directories
mkdir -p "$WORK_DIR/logs"
mkdir -p "$WORK_DIR/models"
mkdir -p "$WORK_DIR/checkpoints"
mkdir -p "$SCRATCH/growai/data"
mkdir -p "$SCRATCH/growai/results"

echo "========================================" | tee -a "$LOG_FILE"
echo "  GrowAI HPC Training Pipeline" | tee -a "$LOG_FILE"
echo "  CINECA Leonardo - aih4a_metafarm" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Budget Overview:" | tee -a "$LOG_FILE"
echo "  Total GPU hours: 10,000" | tee -a "$LOG_FILE"
echo "  Monthly allocation: ~4,000 GPU hours" | tee -a "$LOG_FILE"
echo "  This pipeline: 68 GPU hours total" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check if data exists
if [ ! -f "$SCRATCH/growai/data/sensor_data.csv" ]; then
    echo "⚠️  Warning: Sensor data not found at $SCRATCH/growai/data/sensor_data.csv" | tee -a "$LOG_FILE"
    echo "   Generate synthetic data with: python src/preprocess_data.py --generate" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

# Function to submit job and get ID
submit_job() {
    local job_script=$1
    local job_name=$2
    local gpu_hours=$3
    
    echo "Submitting $job_name..." | tee -a "$LOG_FILE"
    echo "  Script: $job_script" | tee -a "$LOG_FILE"
    echo "  Cost: $gpu_hours GPU hours" | tee -a "$LOG_FILE"
    
    JOB_ID=$(sbatch "$job_script" | awk '{print $4}')
    echo "  Job ID: $JOB_ID" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    echo "$JOB_ID"
}

# Submit jobs with dependencies
echo "========================================" | tee -a "$LOG_FILE"
echo "  Submitting Training Jobs" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 1. Time Series Job (16 GPU hours)
TS_JOB=$(submit_job "scripts/timeseries_job.slurm" "Time Series Training" "16")

# 2. Recommendation Job (4 GPU hours) - can run in parallel
REC_JOB=$(submit_job "scripts/recommendation_job.slurm" "Recommendation Training" "4")

# 3. LLM Job (48 GPU hours) - depends on preprocessing
echo "Submitting LLM Fine-tuning..." | tee -a "$LOG_FILE"
echo "  Note: LLM job can run independently or after other jobs complete" | tee -a "$LOG_FILE"
echo "  Cost: 48 GPU hours" | tee -a "$LOG_FILE"
LLM_JOB=$(sbatch scripts/llm_job.slurm | awk '{print $4}')
echo "  Job ID: $LLM_JOB" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "========================================" | tee -a "$LOG_FILE"
echo "  All Jobs Submitted Successfully!" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Job Summary:" | tee -a "$LOG_FILE"
echo "  Time Series:     $TS_JOB  (16 GPU hours)" | tee -a "$LOG_FILE"
echo "  Recommendation:  $REC_JOB  (4 GPU hours)" | tee -a "$LOG_FILE"
echo "  LLM:             $LLM_JOB  (48 GPU hours)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Total Pipeline Cost: 68 GPU hours" | tee -a "$LOG_FILE"
echo "Remaining Budget: ~9,932 GPU hours" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Monitoring Commands:" | tee -a "$LOG_FILE"
echo "  Check status:   squeue --me" | tee -a "$LOG_FILE"
echo "  Check budget:   saldo -b" | tee -a "$LOG_FILE"
echo "  View logs:      tail -f logs/*.out" | tee -a "$LOG_FILE"
echo "  Cancel jobs:    scancel \$JOB_ID" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results will be saved to:" | tee -a "$LOG_FILE"
echo "  $WORK_DIR/models/" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Completed at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
