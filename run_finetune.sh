#!/bin/bash
#SBATCH -A AIH4A_MetaFarm        # Replace with your account (e.g., tra24_...)
#SBATCH -p boost_usr_prod        # GPU Partition
#SBATCH --time=01:00:00          # 1 Hour
#SBATCH -N 1                     # 1 Node
#SBATCH --gres=gpu:1             # 1 A100 GPU
#SBATCH --mem=64G
#SBATCH --job-name=train_agri
#SBATCH --output=train_log_%j.log

# 1. Load Environment
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.1
source $WORK/vllm-env/bin/activate

# 2. Set Hugging Face Cache to Scratch (Avoids Home quota error)
export HF_HOME=$SCRATCH/huggingface_cache

# 3. Run the Training
python train_agri.py
