# GrowAI HPC Training Pipeline

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.2-green)](https://nvidia.com/cuda)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-ready training pipeline for agricultural AI models on CINECA Leonardo HPC.

## Overview

This repository contains the complete training infrastructure for three agricultural AI models:

1. **Time Series Forecasting Model** - Temporal Fusion Transformer (TFT) for sensor data prediction
2. **Crop Recommendation Model** - XGBoost + Neural Network ensemble for crop recommendations
3. **Agricultural LLM** - QLoRA fine-tuned Qwen 2.5 7B for agricultural Q&A

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Hmauto/growai-hpc.git
cd growai-hpc

# Run complete pipeline
bash run_all.sh

# Or run individual components
bash scripts/setup_env.sh          # Setup environment only
python src/preprocess_data.py       # Preprocess data
sbatch scripts/timeseries_job.slurm # Submit time series training
```

## Repository Structure

```
growai-hpc/
├── scripts/
│   ├── setup_env.sh                # Environment setup
│   ├── timeseries_job.slurm        # Time series SLURM job
│   ├── recommendation_job.slurm    # Recommendation SLURM job
│   └── llm_job.slurm               # LLM fine-tuning SLURM job
├── src/
│   ├── train_timeseries.py         # TFT training script
│   ├── train_recommendation.py     # XGBoost/NN training
│   ├── train_llm.py                # QLoRA fine-tuning
│   └── preprocess_data.py          # Data preprocessing
├── configs/                        # Configuration files
├── data/                          # Data directory (created at runtime)
├── logs/                          # Training logs
├── checkpoints/                   # Model checkpoints
├── run_all.sh                     # Master launcher script
└── README.md                      # This file
```

## Requirements

- CINECA Leonardo account
- Access to `boost_usr_prod` partition
- NVIDIA A100 GPUs (2-4 per job)

## Training Jobs

| Model | Time | GPUs | Memory | Script |
|-------|------|------|--------|--------|
| Time Series | 4 hours | 4x A100 | 120GB | `timeseries_job.slurm` |
| Crop Recommendation | 2 hours | 2x A100 | 64GB | `recommendation_job.slurm` |
| Agricultural LLM | 12 hours | 4x A100 | 200GB | `llm_job.slurm` |

## Detailed Usage

### 1. Environment Setup

```bash
# Interactive setup on login node
bash scripts/setup_env.sh

# This creates a conda environment 'growai-hpc' with:
# - PyTorch 2.1.2 with CUDA 12.2
# - PyTorch Lightning, PyTorch Forecasting
# - XGBoost, scikit-learn
# - Transformers, PEFT, DeepSpeed
# - Flash Attention 2
```

### 2. Data Preprocessing

```bash
# Preprocess all data types
python src/preprocess_data.py

# Or preprocess specific data
python src/preprocess_data.py --only timeseries --timeseries /path/to/raw/sensors.csv
python src/preprocess_data.py --only crop --crop /path/to/raw/crop_data.csv
python src/preprocess_data.py --only llm --llm /path/to/raw/training_data.jsonl
```

### 3. Submit Training Jobs

```bash
# Submit individual jobs
sbatch scripts/timeseries_job.slurm
sbatch scripts/recommendation_job.slurm
sbatch scripts/llm_job.slurm

# Or run complete pipeline
bash run_all.sh

# Monitor jobs
squeue -u $USER
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed
```

### 4. Master Launcher Options

```bash
bash run_all.sh --help

# Options:
#   --setup-only        Only setup environment
#   --preprocess-only   Only run preprocessing
#   --skip-preprocess   Skip preprocessing
#   --skip-timeseries   Skip time series training
#   --skip-recommendation  Skip recommendation training
#   --skip-llm          Skip LLM training
#   --wait              Wait for all jobs to complete
```

## Model Details

### Time Series Forecasting (TFT)

- **Architecture**: Temporal Fusion Transformer
- **Features**: Multi-horizon forecasting with attention mechanisms
- **Input**: Sensor readings (temperature, humidity, soil moisture, etc.)
- **Output**: 24-hour temperature predictions
- **Training**: Multi-GPU with PyTorch Lightning

### Crop Recommendation

- **XGBoost**: Gradient boosted trees with GPU acceleration
- **Neural Network**: Multi-layer perceptron with PyTorch
- **Input**: Soil NPK, pH, temperature, humidity, rainfall
- **Output**: Crop class probabilities
- **Ensemble**: Weighted combination of both models

### Agricultural LLM

- **Base Model**: Qwen/Qwen2.5-7B
- **Method**: QLoRA (4-bit quantization + LoRA)
- **LoRA Config**: r=64, alpha=16, dropout=0.05
- **Training**: DeepSpeed ZeRO-2, multi-GPU
- **Data**: Agricultural Q&A pairs

## Output Structure

```
${WORK}/growai-hpc/
├── checkpoints/
│   ├── timeseries/
│   │   ├── tft-epoch=XX-val_loss=X.XXXX.ckpt
│   │   ├── metrics.json
│   │   └── model_info.json
│   ├── recommendation/
│   │   ├── xgb_model.json
│   │   ├── nn_model.pt
│   │   ├── artifacts.pkl
│   │   └── metrics.json
│   └── llm/
│       ├── checkpoint-XXXX/
│       ├── final/
│       └── merged/
├── logs/
│   ├── timeseries_*.out
│   ├── recommendation_*.out
│   └── llm_*.out
└── data/
    ├── sensors.csv
    ├── crop_data.csv
    └── llm_training.jsonl
```

## Monitoring Training

### TensorBoard

```bash
# On login node
tensorboard --logdir=${WORK}/growai-hpc/logs --port=6006

# Forward port to local machine
ssh -L 6006:localhost:6006 leonardo.cineca.it
```

### Check GPU Usage

```bash
# During job execution
srun --jobid=<job_id> nvidia-smi

# Or attach to running job
sattach <job_id>.0
```

### Check Logs

```bash
# Real-time log monitoring
tail -f ${WORK}/growai-hpc/logs/timeseries_*.out

# After completion
cat ${WORK}/growai-hpc/logs/timeseries_*.out
```

## Troubleshooting

### Common Issues

**Out of Memory**
- Reduce batch size in training scripts
- Use DeepSpeed ZeRO-3 for LLM training
- Enable gradient checkpointing

**NCCL Errors**
- Check network interface: `export NCCL_SOCKET_IFNAME=ib0`
- Verify InfiniBand: `ibstat`

**Job Fails Immediately**
- Check module loading: `module list`
- Verify conda environment: `conda activate growai-hpc`
- Check Python imports: `python -c "import torch; print(torch.__version__)"`

### Getting Help

```bash
# Check job details
scontrol show job <job_id>
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,DerivedExitCode,Comment

# Contact CINECA support
# https://www.cineca.it/en/support
```

## Development

### Adding Custom Data

1. Place raw data in `${SCRATCH}/growai-hpc/data/raw/`
2. Update preprocessing script: `src/preprocess_data.py`
3. Run preprocessing: `python src/preprocess_data.py`

### Modifying Models

1. Edit training script in `src/`
2. Update config in script or create JSON config
3. Test locally with small dataset
4. Submit job with `sbatch`

## License

MIT License - See LICENSE file for details

## Acknowledgments

- CINECA for providing HPC resources
- Leonardo Supercomputer (EuroHPC)
- PyTorch, Hugging Face, and open-source ML communities

## Citation

```bibtex
@software{growai_hpc,
  title = {GrowAI HPC Training Pipeline},
  author = {GrowAI Team},
  year = {2024},
  url = {https://github.com/Hmauto/growai-hpc}
}
```
