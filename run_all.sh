#!/bin/bash
# =============================================================================
# Master Launcher Script for GrowAI HPC Training
# =============================================================================
# This script orchestrates the complete training pipeline on CINECA Leonardo.
# It handles environment setup, data preprocessing, and job submission.
#
# Usage:
#   bash run_all.sh [options]
#
# Options:
#   --setup-only        Only setup environment, don't submit jobs
#   --preprocess-only   Only run preprocessing
#   --skip-preprocess   Skip preprocessing step
#   --skip-timeseries   Skip time series training
#   --skip-recommendation Skip recommendation training
#   --skip-llm          Skip LLM training
#   --wait              Wait for all jobs to complete
#   --help              Show this help message
#
# Examples:
#   bash run_all.sh                    # Run complete pipeline
#   bash run_all.sh --setup-only       # Just setup environment
#   bash run_all.sh --skip-llm         # Run everything except LLM
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK}/growai-hpc"
SCRATCH_DIR="${SCRATCH}/growai-hpc"

# Job IDs storage
declare -A JOB_IDS

# =============================================================================
# Command Line Arguments
# =============================================================================
SETUP_ONLY=false
PREPROCESS_ONLY=false
SKIP_PREPROCESS=false
SKIP_TIMESERIES=false
SKIP_RECOMMENDATION=false
SKIP_LLM=false
WAIT_FOR_COMPLETION=false

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --setup-only)
                SETUP_ONLY=true
                shift
                ;;
            --preprocess-only)
                PREPROCESS_ONLY=true
                shift
                ;;
            --skip-preprocess)
                SKIP_PREPROCESS=true
                shift
                ;;
            --skip-timeseries)
                SKIP_TIMESERIES=true
                shift
                ;;
            --skip-recommendation)
                SKIP_RECOMMENDATION=true
                shift
                ;;
            --skip-llm)
                SKIP_LLM=true
                shift
                ;;
            --wait)
                WAIT_FOR_COMPLETION=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    head -n 40 "${BASH_SOURCE[0]}" | tail -n 35
}

# =============================================================================
# Logging
# =============================================================================
LOG_FILE="${WORK_DIR}/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE" >&2
}

# =============================================================================
# Environment Setup
# =============================================================================
setup_environment() {
    log "========================================"
    log "Setting up environment..."
    log "========================================"
    
    # Create directories
    mkdir -p "${WORK_DIR}"/{scripts,src,configs,data,logs,checkpoints}
    mkdir -p "${SCRATCH_DIR}"/{data,checkpoints,temp,cache}
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Check if environment exists
    if conda env list | grep -q "^growai-hpc"; then
        log "Environment 'growai-hpc' already exists. Skipping setup."
        log "To recreate, run: conda env remove -n growai-hpc && bash scripts/setup_env.sh"
        return 0
    fi
    
    # Run setup script
    if [ -f "${SCRIPT_DIR}/setup_env.sh" ]; then
        log "Running setup_env.sh..."
        bash "${SCRIPT_DIR}/setup_env.sh"
    else
        error "setup_env.sh not found in ${SCRIPT_DIR}"
        return 1
    fi
    
    log "Environment setup complete!"
}

# =============================================================================
# Data Preprocessing
# =============================================================================
run_preprocessing() {
    log "========================================"
    log "Running data preprocessing..."
    log "========================================"
    
    # Load modules
    module purge
    module load profile/deeplearning
    source "${CONDA_DIR}/etc/profile.d/conda.sh" 2>/dev/null || \
    source "${HOME}/.conda/etc/profile.d/conda.sh" 2>/dev/null || \
    module load anaconda3
    conda activate growai-hpc
    
    # Run preprocessing script
    python "${WORK_DIR}/src/preprocess_data.py" \
        --output-dir "${SCRATCH_DIR}/data" \
        2>&1 | tee -a "$LOG_FILE"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        log "Preprocessing complete!"
    else
        error "Preprocessing failed with exit code: $exit_code"
        return 1
    fi
}

# =============================================================================
# Job Submission
# =============================================================================
submit_job() {
    local job_name=$1
    local script_name=$2
    local dependencies=$3
    
    log "Submitting ${job_name} job..."
    
    local submit_cmd="sbatch"
    if [ -n "$dependencies" ]; then
        submit_cmd="${submit_cmd} --dependency=afterok:${dependencies}"
    fi
    
    local output
    output=$(${submit_cmd} "${SCRIPT_DIR}/${script_name}" 2>&1)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        local job_id
        job_id=$(echo "$output" | grep -oP '\d+')
        JOB_IDS[$job_name]=$job_id
        log "✓ ${job_name} job submitted: ${job_id}"
        echo "$job_id"
    else
        error "Failed to submit ${job_name} job: $output"
        return 1
    fi
}

# =============================================================================
# Monitor Jobs
# =============================================================================
wait_for_jobs() {
    log "========================================"
    log "Waiting for jobs to complete..."
    log "========================================"
    
    local all_completed=false
    local check_interval=60  # seconds
    
    while [ "$all_completed" = false ]; do
        all_completed=true
        
        for job_name in "${!JOB_IDS[@]}"; do
            local job_id=${JOB_IDS[$job_name]}
            local status
            status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null || echo "COMPLETED")
            
            if [ "$status" = "COMPLETED" ] || [ "$status" = "" ]; then
                log "  ${job_name} (${job_id}): COMPLETED"
            elif [ "$status" = "FAILED" ]; then
                log "  ${job_name} (${job_id}): FAILED"
            elif [ "$status" = "CANCELLED" ]; then
                log "  ${job_name} (${job_id}): CANCELLED"
            else
                log "  ${job_name} (${job_id}): ${status}"
                all_completed=false
            fi
        done
        
        if [ "$all_completed" = false ]; then
            log "Checking again in ${check_interval} seconds..."
            sleep $check_interval
        fi
    done
    
    log "All jobs completed!"
}

# =============================================================================
# Main Pipeline
# =============================================================================
main() {
    parse_args "$@"
    
    echo "========================================"
    echo "GrowAI HPC Training Pipeline"
    echo "========================================"
    echo "Started at: $(date)"
    echo "Log file: ${LOG_FILE}"
    echo "========================================"
    
    # Setup environment
    setup_environment
    
    if [ "$SETUP_ONLY" = true ]; then
        log "Setup complete. Exiting as requested."
        exit 0
    fi
    
    # Data preprocessing
    if [ "$SKIP_PREPROCESS" = false ]; then
        run_preprocessing
        
        if [ "$PREPROCESS_ONLY" = true ]; then
            log "Preprocessing complete. Exiting as requested."
            exit 0
        fi
    else
        log "Skipping preprocessing as requested."
    fi
    
    # Submit training jobs
    log "========================================"
    log "Submitting training jobs..."
    log "========================================"
    
    local dependencies=""
    
    # Time Series Job (4 hours, 4x A100)
    if [ "$SKIP_TIMESERIES" = false ]; then
        JOB_IDS["timeseries"]=$(submit_job "timeseries" "timeseries_job.slurm" "")
        dependencies="${dependencies},${JOB_IDS["timeseries"]}"
    fi
    
    # Recommendation Job (2 hours, 2x A100) - can run in parallel
    if [ "$SKIP_RECOMMENDATION" = false ]; then
        JOB_IDS["recommendation"]=$(submit_job "recommendation" "recommendation_job.slurm" "")
        dependencies="${dependencies},${JOB_IDS["recommendation"]}"
    fi
    
    # LLM Job (12 hours, 4x A100) - depends on preprocessing but can run in parallel to other trainings
    if [ "$SKIP_LLM" = false ]; then
        # LLM can run in parallel since preprocessing is done
        JOB_IDS["llm"]=$(submit_job "llm" "llm_job.slurm" "")
        dependencies="${dependencies},${JOB_IDS["llm"]}"
    fi
    
    # Remove leading comma from dependencies
    dependencies="${dependencies#,}"
    
    log "========================================"
    log "Job Summary"
    log "========================================"
    for job_name in "${!JOB_IDS[@]}"; do
        log "  ${job_name}: ${JOB_IDS[$job_name]}"
    done
    log "========================================"
    log "Monitor jobs with: squeue -u \$USER"
    log "Cancel all jobs: scancel -u \$USER -n 'growai-*'"
    log "========================================"
    
    # Wait for completion if requested
    if [ "$WAIT_FOR_COMPLETION" = true ]; then
        wait_for_jobs
        
        log "========================================"
        log "Training pipeline complete!"
        log "========================================"
        log "Checkpoints: ${WORK_DIR}/checkpoints/"
        log "Logs: ${WORK_DIR}/logs/"
        log "========================================"
    else
        log "Jobs submitted. Use --wait flag to monitor completion."
    fi
}

# =============================================================================
# Run Main
# =============================================================================
main "$@"
