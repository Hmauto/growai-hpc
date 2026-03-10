#!/bin/bash
###############################################################################
# Grow AI - Master Setup Script for CINECA Leonardo HPC
# This script sets up the complete training environment for Grow AI models
###############################################################################

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           Grow AI - CINECA Leonardo Setup                        ║"
echo "║     Time Series + Recommendation + LLM Training                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
GROWAI_DIR="$HOME/growai"
DATA_DIR="$GROWAI_DIR/data"
MODELS_DIR="$GROWAI_DIR/models"
SCRIPTS_DIR="$GROWAI_DIR/scripts"
RESULTS_DIR="$GROWAI_DIR/results"
SCRATCH_DIR="$SCRATCH/growai"

echo -e "${BLUE}[1/6] Creating directory structure...${NC}"
mkdir -p $GROWAI_DIR/{data,models,scripts,results}
mkdir -p $DATA_DIR/{raw,processed,train,val,test}
mkdir -p $SCRATCH_DIR
echo -e "${GREEN}✓ Directories created${NC}"

echo ""
echo -e "${BLUE}[2/6] Loading CINECA modules...${NC}"
module purge
module load python/3.11.7 || module load python/3.10.9
module load profile/deeplrn
module load cuda/11.8
module list
echo -e "${GREEN}✓ Modules loaded${NC}"

echo ""
echo -e "${BLUE}[3/6] Creating Python virtual environment...${NC}"
cd $GROWAI_DIR
python3 -m venv venv
source venv/bin/activate

echo -e "${BLUE}[4/6] Installing Python packages...${NC}"
pip install --upgrade pip

# Core ML/DL packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning

# Time Series Forecasting
pip install pytorch-forecasting pandas numpy scikit-learn

# Recommendation Models
pip install xgboost lightgbm catboost

# LLM Fine-tuning
pip install transformers datasets accelerate peft trl bitsandbytes

# Utilities
pip install matplotlib seaborn tensorboard jupyter tqdm python-dotenv

echo -e "${GREEN}✓ Python environment ready${NC}"

echo ""
echo -e "${BLUE}[5/6] Creating training scripts...${NC}"

cat > $SCRIPTS_DIR/train_timeseries.py << 'PYEOF'
#!/usr/bin/env python3
"""
Time Series Forecasting Model for Grow AI
Trains Temporal Fusion Transformer on sensor data
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, MAE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_path}")
    data = pd.read_csv(args.data_path, parse_dates=['timestamp'])
    
    # Create time features
    data['time_idx'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds() / 3600
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['month'] = data['timestamp'].dt.month
    
    print(f"Data shape: {data.shape}")
    print(f"Fields: {data['field_id'].nunique()}")
    
    # Create dataset
    max_encoder_length = 168  # 7 days
    max_prediction_length = 72  # 3 days
    
    training = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="moisture",
        group_ids=["field_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["crop_type"],
        time_varying_known_reals=["hour", "day_of_week", "month"],
        time_varying_unknown_reals=["moisture", "temperature", "humidity", "rainfall", "n", "p", "k", "ph"],
        target_normalizer=GroupNormalizer(groups=["field_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create dataloaders
    train_dataloader = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=8)
    val_dataloader = training.to_dataloader(train=False, batch_size=args.batch_size, num_workers=8)
    
    # Create model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=160,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=80,
        output_size=7,
        loss=SMAPE(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in tft.parameters())}")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=-1,
        strategy='ddp',
        gradient_clip_val=0.1,
        limit_train_batches=1.0,
        callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
    )
    
    # Train
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(tft.state_dict(), os.path.join(args.output_dir, 'timeseries_model.pth'))
    print(f"Model saved to {args.output_dir}")

if __name__ == '__main__':
    main()
PYEOF

cat > $SCRIPTS_DIR/train_recommendation.py << 'PYEOF'
#!/usr/bin/env python3
"""
Crop Recommendation Model for Grow AI
Trains XGBoost + Neural Network for agricultural recommendations
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

class CropDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RecommendationNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def train_xgboost(X_train, y_train, X_val, y_val, output_dir):
    """Train XGBoost model"""
    print("Training XGBoost model...")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=True
    )
    
    # Save
    model.save_model(os.path.join(output_dir, 'recommendation_xgb.json'))
    
    # Evaluate
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"XGBoost Val Accuracy: {acc:.4f}")
    
    return model

def train_neural_net(X_train, y_train, X_val, y_val, input_dim, output_dim, output_dir):
    """Train Neural Network"""
    print("Training Neural Network...")
    
    train_dataset = CropDataset(X_train, y_train)
    val_dataset = CropDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RecommendationNet(input_dim, 256, output_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_acc = 0
    for epoch in range(50):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        acc = correct / total
        scheduler.step(train_loss)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'recommendation_nn.pth'))
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, Acc={acc:.4f}")
    
    print(f"Best NN Val Accuracy: {best_acc:.4f}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--model', type=str, default='both', choices=['xgboost', 'nn', 'both'])
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    # Features and target
    feature_cols = ['moisture', 'temperature', 'humidity', 'rainfall', 'n', 'p', 'k', 'ph', 'ec']
    target_col = 'recommended_action'
    
    X = df[feature_cols].values
    y = LabelEncoder().fit_transform(df[target_col].values)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train models
    if args.model in ['xgboost', 'both']:
        train_xgboost(X_train, y_train, X_val, y_val, args.output_dir)
    
    if args.model in ['nn', 'both']:
        train_neural_net(X_train, y_train, X_val, y_val, X.shape[1], len(np.unique(y)), args.output_dir)
    
    print(f"Models saved to {args.output_dir}")

if __name__ == '__main__':
    main()
PYEOF

cat > $SCRIPTS_DIR/train_llm.py << 'PYEOF'
#!/usr/bin/env python3
"""
Agricultural LLM Fine-tuning for Grow AI
Fine-tunes Qwen 2.5 with QLoRA for farming advice
"""

import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json

def format_prompt(example):
    """Format training example"""
    system_msg = "You are Grow AI, an expert agricultural advisor. Explain sensor data simply to farmers."
    
    prompt = f"""### System:
{system_msg}

### Field Data:
- Crop: {example['crop_type']}
- Soil Moisture: {example['moisture']}%
- Temperature: {example['temperature']}°C
- Humidity: {example['humidity']}%
- NPK: {example['n']}/{example['p']}/{example['k']} mg/kg
- pH: {example['ph']}

### Recommendation:
{example['recommendation']}

### Advice:
"""
    
    return {"text": prompt + example['explanation']}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    
    print(f"Loading base model: {args.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )
    
    model = get_peft_model(model, lora_config)
    print(f"Trainable parameters: {model.print_trainable_parameters()}")
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    with open(args.data_path, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    dataset = Dataset.from_list(examples)
    dataset = dataset.map(format_prompt)
    
    # Tokenize
    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, max_length=2048, padding='max_length')
    
    dataset = dataset.map(tokenize, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        optim='paged_adamw_8bit',
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Trainer
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save
    model.save_pretrained(os.path.join(args.output_dir, 'final_model'))
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'final_model'))
    print(f"Model saved to {args.output_dir}")

if __name__ == '__main__':
    main()
PYEOF

echo -e "${GREEN}✓ Training scripts created${NC}"

echo ""
echo -e "${BLUE}[6/6] Creating SLURM job scripts...${NC}"

# Time Series Job
cat > $SCRIPTS_DIR/timeseries_job.slurm << 'SLURMEOF'
#!/bin/bash
#SBATCH --job-name=growai_timeseries
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120GB
#SBATCH --output=timeseries_%j.out

echo "Grow AI - Time Series Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

module load python/3.11.7 profile/deeplrn cuda/11.8
source $HOME/growai/venv/bin/activate

python $HOME/growai/scripts/train_timeseries.py \
    --data-path $SCRATCH/growai/data/sensor_data.csv \
    --output-dir $SCRATCH/growai/results/timeseries_$SLURM_JOB_ID \
    --max-epochs 50 \
    --batch-size 64

echo "Completed: $(date)"
SLURMEOF

# Recommendation Job
cat > $SCRIPTS_DIR/recommendation_job.slurm << 'SLURMEOF'
#!/bin/bash
#SBATCH --job-name=growai_recommendation
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --output=recommendation_%j.out

echo "Grow AI - Recommendation Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

module load python/3.11.7 profile/deeplrn cuda/11.8
source $HOME/growai/venv/bin/activate

python $HOME/growai/scripts/train_recommendation.py \
    --data-path $SCRATCH/growai/data/training_data.csv \
    --output-dir $SCRATCH/growai/results/recommendation_$SLURM_JOB_ID \
    --model both

echo "Completed: $(date)"
SLURMEOF

# LLM Job
cat > $SCRIPTS_DIR/llm_job.slurm << 'SLURMEOF'
#!/bin/bash
#SBATCH --job-name=growai_llm
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240GB
#SBATCH --output=llm_%j.out

echo "Grow AI - LLM Fine-tuning"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

module load python/3.11.7 profile/deeplrn cuda/11.8
source $HOME/growai/venv/bin/activate

python $HOME/growai/scripts/train_llm.py \
    --data-path $SCRATCH/growai/data/llm_training.jsonl \
    --output-dir $SCRATCH/growai/results/llm_$SLURM_JOB_ID \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --epochs 3 \
    --batch-size 4

echo "Completed: $(date)"
SLURMEOF

# Master launcher
cat > $GROWAI_DIR/run_all.sh << 'LAUNCHEOF'
#!/bin/bash
# Master launcher for all Grow AI training jobs

echo "Launching Grow AI Training Jobs..."
echo ""

# Submit jobs
echo "Submitting Time Series job..."
TS_JOB=$(sbatch $HOME/growai/scripts/timeseries_job.slurm | awk '{print $4}')
echo "  Job ID: $TS_JOB"

echo "Submitting Recommendation job..."
REC_JOB=$(sbatch $HOME/growai/scripts/recommendation_job.slurm | awk '{print $4}')
echo "  Job ID: $REC_JOB"

echo "Submitting LLM job..."
LLM_JOB=$(sbatch $HOME/growai/scripts/llm_job.slurm | awk '{print $4}')
echo "  Job ID: $LLM_JOB"

echo ""
echo "All jobs submitted!"
echo "Monitor with: squeue --me"
echo ""
echo "Job IDs:"
echo "  Time Series: $TS_JOB"
echo "  Recommendation: $REC_JOB"
echo "  LLM: $LLM_JOB"
LAUNCHEOF

chmod +x $GROWAI_DIR/run_all.sh
chmod +x $SCRIPTS_DIR/*.py
chmod +x $SCRIPTS_DIR/*.slurm

echo -e "${GREEN}✓ SLURM job scripts created${NC}"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo -e "${GREEN}              ✅ Setup Complete!                                  ${NC}"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Directory structure:"
echo "  $GROWAI_DIR/"
echo "  ├── venv/              # Python environment"
echo "  ├── data/              # Datasets"
echo "  ├── models/            # Saved models"
echo "  ├── scripts/           # Training scripts"
echo "  └── results/           # Training results"
echo ""
echo "Training scripts:"
echo "  • train_timeseries.py      - TFT for sensor forecasting"
echo "  • train_recommendation.py  - XGBoost + Neural Network"
echo "  • train_llm.py            - QLoRA fine-tuning"
echo ""
echo "SLURM jobs:"
echo "  • timeseries_job.slurm     - 4 hours, 4x A100"
echo "  • recommendation_job.slurm - 2 hours, 2x A100"
echo "  • llm_job.slurm           - 12 hours, 4x A100"
echo ""
echo "To run all: ./run_all.sh"
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "  Next: Upload data to $SCRATCH/growai/data/"
echo "  Then: Submit jobs with ./run_all.sh"
echo "╚══════════════════════════════════════════════════════════════════╝"
