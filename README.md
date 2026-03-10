# Grow AI - HPC Training Pipeline for CINECA Leonardo

Automated training pipeline for Grow AI agricultural intelligence models on CINECA Leonardo HPC.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/Hmauto/growai-hpc.git
cd growai-hpc
bash setup_growai.sh

# Upload your data to $SCRATCH/growai/data/
# Then run all training jobs
./run_all.sh
```

## 📁 What's Included

### Training Scripts
- `train_timeseries.py` - Temporal Fusion Transformer for sensor forecasting
- `train_recommendation.py` - XGBoost + Neural Network for crop recommendations
- `train_llm.py` - QLoRA fine-tuning of Qwen 2.5 for agricultural advice

### SLURM Jobs
- `timeseries_job.slurm` - 4 hours, 4x A100 GPUs
- `recommendation_job.slurm` - 2 hours, 2x A100 GPUs
- `llm_job.slurm` - 12 hours, 4x A100 GPUs

### Setup
- `setup_growai.sh` - Complete environment setup
- `run_all.sh` - Master launcher for all jobs

## 🎯 Models

### 1. Time Series Forecaster
**Purpose**: Predict soil moisture, weather trends
- **Architecture**: Temporal Fusion Transformer
- **Input**: 7 days of sensor history
- **Output**: 3-day forecast (moisture, stress probability)
- **Training**: ~4 hours on 4x A100

### 2. Recommendation Engine
**Purpose**: Recommend irrigation, fertilization actions
- **Architecture**: XGBoost + Neural Network
- **Input**: Current sensor readings
- **Output**: Recommended action + confidence
- **Training**: ~2 hours on 2x A100

### 3. Agricultural LLM
**Purpose**: Explain recommendations in natural language
- **Base Model**: Qwen 2.5 7B Instruct
- **Method**: QLoRA fine-tuning (4-bit)
- **Training**: ~12 hours on 4x A100
- **Languages**: English, French, Arabic (Darija)

## 📊 Data Format

### Sensor Data (CSV)
```csv
timestamp,field_id,moisture,temperature,humidity,rainfall,n,p,k,ph,ec,crop_type
2024-01-01 00:00,F001,45.2,22.1,65,0,120,45,180,6.5,1.2,tomato
```

### LLM Training (JSONL)
```json
{
  "crop_type": "tomato",
  "moisture": 25,
  "temperature": 30,
  "recommendation": "irrigate_20mm",
  "explanation": "Your soil is dry. Water your tomatoes with 20mm today..."
}
```

## 🖥️ CINECA Configuration

**System**: Leonardo  
**Partition**: boost_usr_prod  
**GPUs**: 4x NVIDIA A100 64GB  
**Documentation**: https://docs.hpc.cineca.it/

## 📈 Training Times

| Model | GPUs | Time | Output |
|-------|------|------|--------|
| Time Series | 4x A100 | 4h | `timeseries_model.pth` |
| Recommendation | 2x A100 | 2h | `recommendation_*.json/pth` |
| LLM | 4x A100 | 12h | `final_model/` |

## 🔧 Usage

### Setup (One time)
```bash
bash setup_growai.sh
```

### Upload Data
```bash
# On your local machine
scp sensor_data.csv user@login.leonardo.cineca.it:$SCRATCH/growai/data/
```

### Run Training
```bash
# All models
./run_all.sh

# Or individually
sbatch scripts/timeseries_job.slurm
sbatch scripts/recommendation_job.slurm
sbatch scripts/llm_job.slurm
```

### Monitor
```bash
squeue --me
tail -f timeseries_*.out
```

### Download Results
```bash
scp -r user@login.leonardo.cineca.it:$SCRATCH/growai/results ./
```

## 📚 Files

```
growai-hpc/
├── setup_growai.sh              # Main setup script
├── run_all.sh                   # Launch all jobs
├── README.md                    # This file
└── scripts/
    ├── train_timeseries.py      # TFT training
    ├── train_recommendation.py  # XGBoost/NN training
    ├── train_llm.py            # LLM fine-tuning
    ├── timeseries_job.slurm     # SLURM: Time Series
    ├── recommendation_job.slurm # SLURM: Recommendation
    └── llm_job.slurm           # SLURM: LLM
```

## 🌱 Integration with MetaFarm

```
MetaFarm (Vision)     +     Grow AI (Sensors + LLM)
       |                           |
   Disease                     Soil/Weather
   Detection                   Forecasting
       |                           |
       └───────────┬───────────────┘
                   |
            Unified API
                   |
              WhatsApp Bot
           (Farmer Interface)
```

## ⚙️ Requirements

- CINECA Leonardo access
- `aih4a_metafarm` account (or adjust time limits in SLURM scripts)
- Data in `$SCRATCH/growai/data/`

## 📝 Notes

- Adjust SLURM time limits based on your account quota
- LLM training uses 4-bit quantization to fit on 4x A100
- All models support multi-GPU training (DDP)

## 📞 Support

- CINECA Help: superc@cineca.it
- Documentation: https://docs.hpc.cineca.it/

---
**Ready to train! Run `bash setup_growai.sh` on CINECA** 🚀
