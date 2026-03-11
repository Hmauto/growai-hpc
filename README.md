# Grow AI - HPC Training Pipeline for CINECA Leonardo

Automated training pipeline for Grow AI agricultural intelligence models on CINECA Leonardo HPC.

## 🎯 CINECA Resource Allocation

- **Project**: aih4a_metafarm
- **Budget**: 10,000 GPU hours
- **Monthly**: ~4,000 GPU hours
- **Storage**: 1 TB ($WORK)
- **Project End**: March 31, 2026

## 🚀 Quick Start

```bash
# Clone on CINECA Leonardo
git clone https://github.com/Hmauto/growai-hpc.git
cd growai-hpc

# Run complete pipeline (68 GPU hours total)
bash run_all.sh
```

## 📁 Files

| File | Purpose | GPU Hours |
|------|---------|-----------|
| `scripts/timeseries_job.slurm` | TFT forecasting | 16 |
| `scripts/recommendation_job.slurm` | XGBoost + NN | 4 |
| `scripts/llm_job.slurm` | LLM fine-tuning | 48 |
| `run_all.sh` | Launch all jobs | 68 total |

## 💰 Budget Breakdown

| Job | GPUs | Hours | Cost |
|-----|------|-------|------|
| Time Series | 4 | 4 | 16 GPU hours |
| Recommendation | 2 | 2 | 4 GPU hours |
| LLM | 4 | 12 | 48 GPU hours |
| **Total** | | | **68 GPU hours** |

Can run ~147 full pipelines within budget.

## 📊 Check Budget

```bash
saldo -b
```

## 🖥️ CINECA Configuration

- **Partition**: boost_usr_prod
- **Account**: aih4a_metafarm
- **Documentation**: https://docs.hpc.cineca.it/

---
