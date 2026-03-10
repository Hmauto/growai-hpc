#!/bin/bash
# =============================================================================
# CINECA Leonardo HPC Environment Setup Script
# =============================================================================
# This script sets up the conda environment and installs all dependencies
# for training three agricultural AI models on CINECA Leonardo.
#
# Usage: sbatch setup_env.sh
#        or: bash setup_env.sh (for interactive login nodes)
#
# Author: GrowAI Team
# Date: 2024
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
ENV_NAME="growai-hpc"
PYTHON_VERSION="3.10"
WORK_DIR="${WORK}/growai-hpc"
SCRATCH_DIR="${SCRATCH}/growai-hpc"

# =============================================================================
# Logging
# =============================================================================
LOG_FILE="${WORK_DIR}/logs/setup_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE" >&2
    exit 1
}

# =============================================================================
# Load CINECA Modules
# =============================================================================
log "Loading CINECA Leonardo modules..."

module purge
module load profile/deeplearning
module load cuda/12.2
module load cudnn/8.9.7.29-cuda-12.2
module load nccl/2.18.5-cuda-12.2
module load openmpi/4.1.6--gcc--12.2.0
module load gcc/12.2.0

# Verify modules loaded
log "Loaded modules:"
module list 2>&1 | tee -a "$LOG_FILE"

# =============================================================================
# Create Working Directories
# =============================================================================
log "Creating working directories..."

mkdir -p "${WORK_DIR}"/{scripts,src,configs,data,logs,checkpoints,results}
mkdir -p "${SCRATCH_DIR}"/{data,checkpoints,temp}

log "Work directory: ${WORK_DIR}"
log "Scratch directory: ${SCRATCH_DIR}"

# =============================================================================
# Setup Conda Environment
# =============================================================================
log "Setting up conda environment: ${ENV_NAME}"

# Source conda
source "${CONDA_DIR}/etc/profile.d/conda.sh" 2>/dev/null || \
source "${HOME}/.conda/etc/profile.d/conda.sh" 2>/dev/null || \
source "/opt/conda/etc/profile.d/conda.sh" 2>/dev/null || \
{ log "Conda not found in standard locations, trying module..."; module load anaconda3; }

# Remove existing environment if present
if conda env list | grep -q "^${ENV_NAME}"; then
    log "Removing existing environment ${ENV_NAME}..."
    conda env remove -n "${ENV_NAME}" -y
fi

# Create new environment
log "Creating new conda environment with Python ${PYTHON_VERSION}..."
conda create -n "${ENV_NAME}" python=${PYTHON_VERSION} -y

# Activate environment
conda activate "${ENV_NAME}"
log "Activated environment: ${ENV_NAME}"

# =============================================================================
# Install PyTorch (CUDA 12.2)
# =============================================================================
log "Installing PyTorch with CUDA 12.2 support..."

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch installation
python -c "import torch; log(f'PyTorch version: {torch.__version__}'); log(f'CUDA available: {torch.cuda.is_available()}'); log(f'CUDA version: {torch.version.cuda}')" || error "PyTorch installation failed"

# =============================================================================
# Install Core ML Libraries
# =============================================================================
log "Installing core machine learning libraries..."

# Time Series & Data Processing
pip install \
    pytorch-forecasting==1.0.0 \
    pytorch-lightning==2.1.3 \
    pandas==2.1.4 \
    numpy==1.26.3 \
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    prophet==1.1.5 \
    statsmodels==0.14.1

# Deep Learning & Transformers
pip install \
    transformers==4.36.2 \
    peft==0.7.1 \
    bitsandbytes==0.41.3 \
    accelerate==0.25.0 \
    datasets==2.16.1 \
    tokenizers==0.15.0 \
    sentencepiece==0.1.99 \
    protobuf==4.25.1

# Distributed Training
pip install \
    deepspeed==0.12.6 \
    wandb==0.16.2 \
    tensorboard==2.15.1

# Utilities
pip install \
    tqdm==4.66.1 \
    pyyaml==6.0.1 \
    python-dotenv==1.0.0 \
    requests==2.31.0 \
    pyarrow==14.0.2 \
    fastparquet==2023.10.1

# =============================================================================
# Install Flash Attention (Optional but Recommended)
# =============================================================================
log "Installing Flash Attention 2..."
pip install flash-attn==2.4.2 --no-build-isolation || log "Flash Attention installation skipped (optional)"

# =============================================================================
# Verify Installations
# =============================================================================
log "Verifying installations..."

python << 'EOF'
import sys

def check_import(module_name, package_name=None):
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name}: {e}")
        return False

all_ok = True
packages = [
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "XGBoost"),
    ("pytorch_forecasting", "PyTorch Forecasting"),
    ("pytorch_lightning", "PyTorch Lightning"),
    ("transformers", "Transformers"),
    ("peft", "PEFT"),
    ("bitsandbytes", "BitsAndBytes"),
    ("accelerate", "Accelerate"),
    ("datasets", "Datasets"),
    ("deepspeed", "DeepSpeed"),
    ("wandb", "Weights & Biases"),
]

print("\nPackage Status:")
print("-" * 40)
for module, package in packages:
    if not check_import(module, package):
        all_ok = False

print("-" * 40)
if all_ok:
    print("✓ All packages installed successfully!")
    sys.exit(0)
else:
    print("✗ Some packages failed to install")
    sys.exit(1)
EOF

# =============================================================================
# Create Activation Script
# =============================================================================
log "Creating environment activation script..."

cat > "${WORK_DIR}/activate_env.sh" << 'EOF'
#!/bin/bash
# Activate the GrowAI HPC environment

module purge
module load profile/deeplearning
module load cuda/12.2
module load cudnn/8.9.7.29-cuda-12.2
module load nccl/2.18.5-cuda-12.2
module load openmpi/4.1.6--gcc--12.2.0
module load gcc/12.2.0

source "${CONDA_DIR}/etc/profile.d/conda.sh" 2>/dev/null || \
source "${HOME}/.conda/etc/profile.d/conda.sh" 2>/dev/null || \
source "/opt/conda/etc/profile.d/conda.sh" 2>/dev/null || \
module load anaconda3

conda activate growai-hpc

echo "Environment activated: growai-hpc"
echo "Python: $(which python)"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else \"N/A\")')"
EOF

chmod +x "${WORK_DIR}/activate_env.sh"

# =============================================================================
# Summary
# =============================================================================
log "========================================"
log "Setup completed successfully!"
log "========================================"
log "Environment: ${ENV_NAME}"
log "Work Directory: ${WORK_DIR}"
log "Scratch Directory: ${SCRATCH_DIR}"
log "Log File: ${LOG_FILE}"
log ""
log "To activate the environment, run:"
log "  source ${WORK_DIR}/activate_env.sh"
log ""
log "To submit training jobs:"
log "  cd ${WORK_DIR}/scripts"
log "  sbatch timeseries_job.slurm"
log "  sbatch recommendation_job.slurm"
log "  sbatch llm_job.slurm"
log "========================================"
