#!/usr/bin/env python3
"""
Time Series Forecasting Model Training Script
==============================================
Trains a Temporal Fusion Transformer (TFT) for agricultural sensor data forecasting.
Optimized for CINECA Leonardo HPC with multi-GPU support.

Author: GrowAI Team
Date: 2024
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# PyTorch Forecasting
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
    GroupNormalizer,
    QuantileLoss
)
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Data
    'data_path': os.environ.get('SCRATCH', '.') + '/growai-hpc/data/sensors.csv',
    'max_sequence_length': 168,  # 1 week of hourly data
    'max_prediction_length': 24,  # Predict next 24 hours
    'max_encoder_length': 168,
    
    # Model
    'hidden_size': 160,
    'lstm_layers': 2,
    'num_attention_heads': 4,
    'dropout': 0.1,
    'hidden_continuous_size': 80,
    'attention_head_size': 40,
    
    # Training
    'batch_size': 128,
    'learning_rate': 0.001,
    'max_epochs': 100,
    'gradient_clip_val': 0.1,
    'early_stopping_patience': 10,
    
    # Output
    'checkpoint_dir': os.environ.get('WORK', '.') + '/growai-hpc/checkpoints/timeseries',
    'log_dir': os.environ.get('WORK', '.') + '/growai-hpc/logs/timeseries',
    
    # Distributed
    'gpus': -1,  # Use all available GPUs
    'strategy': 'ddp',
}

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('timeseries_training')
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = os.path.join(log_dir, f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# =============================================================================
# Data Loading & Preprocessing
# =============================================================================

def load_sensor_data(data_path: str) -> pd.DataFrame:
    """Load and validate sensor data."""
    logger = logging.getLogger('timeseries_training')
    
    if not os.path.exists(data_path):
        logger.warning(f"Data file not found: {data_path}")
        logger.info("Generating synthetic sensor data for testing...")
        return generate_synthetic_data()
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    return df


def generate_synthetic_data(n_samples: int = 100000) -> pd.DataFrame:
    """Generate synthetic agricultural sensor data for testing."""
    np.random.seed(42)
    
    # Create time series for multiple sensors
    sensors = [f'sensor_{i:03d}' for i in range(50)]
    dates = pd.date_range('2023-01-01', periods=n_samples // len(sensors), freq='H')
    
    data = []
    for sensor_id in sensors:
        base_temp = np.random.uniform(15, 30)
        base_humidity = np.random.uniform(40, 80)
        base_soil = np.random.uniform(20, 60)
        
        for i, date in enumerate(dates):
            # Add seasonality and noise
            hour_factor = np.sin(2 * np.pi * date.hour / 24)
            day_factor = np.sin(2 * np.pi * date.dayofyear / 365)
            
            data.append({
                'time_idx': i,
                'sensor_id': sensor_id,
                'timestamp': date,
                'temperature': base_temp + 5 * hour_factor + 10 * day_factor + np.random.normal(0, 2),
                'humidity': base_humidity + 10 * hour_factor + np.random.normal(0, 5),
                'soil_moisture': base_soil + 15 * np.sin(2 * np.pi * i / 168) + np.random.normal(0, 3),
                'light_intensity': max(0, 500 * np.sin(2 * np.pi * date.hour / 24) + np.random.normal(0, 50)),
                'co2_level': 400 + 100 * np.random.random(),
            })
    
    df = pd.DataFrame(data)
    
    # Add target variable (future temperature)
    df = df.sort_values(['sensor_id', 'time_idx'])
    df['target_temperature'] = df.groupby('sensor_id')['temperature'].shift(-24)
    df = df.dropna()
    
    return df


def preprocess_data(df: pd.DataFrame, config: Dict) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Preprocess data for TFT model."""
    logger = logging.getLogger('timeseries_training')
    
    # Ensure correct dtypes
    df['sensor_id'] = df['sensor_id'].astype(str)
    df['time_idx'] = df['time_idx'].astype(int)
    
    # Add time features
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour.astype(str)
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek.astype(str)
        df['month'] = pd.to_datetime(df['timestamp']).dt.month.astype(str)
    
    # Define features
    time_varying_known_categoricals = ['hour', 'day_of_week', 'month']
    time_varying_known_reals = ['time_idx']
    
    time_varying_unknown_categoricals = []
    time_varying_unknown_reals = [
        'temperature', 'humidity', 'soil_moisture', 
        'light_intensity', 'co2_level'
    ]
    
    static_categoricals = ['sensor_id']
    
    # Create dataset
    max_encoder_length = config['max_encoder_length']
    max_prediction_length = config['max_prediction_length']
    
    # Training cutoff
    training_cutoff = df['time_idx'].max() - max_prediction_length * 2
    
    logger.info(f"Training cutoff: {training_cutoff}")
    logger.info(f"Encoder length: {max_encoder_length}")
    logger.info(f"Prediction length: {max_prediction_length}")
    
    # Training dataset
    training = TimeSeriesDataSet(
        df[df['time_idx'] <= training_cutoff],
        time_idx='time_idx',
        target='target_temperature' if 'target_temperature' in df.columns else 'temperature',
        group_ids=['sensor_id'],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=['sensor_id'],
            transformation='softplus'
        ),
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    
    # Validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        predict=True,
        stop_randomization=True
    )
    
    logger.info(f"Training samples: {len(training)}")
    logger.info(f"Validation samples: {len(validation)}")
    
    return training, validation


# =============================================================================
# Model Definition
# =============================================================================

def create_tft_model(training_dataset: TimeSeriesDataSet, config: Dict) -> TemporalFusionTransformer:
    """Create Temporal Fusion Transformer model."""
    logger = logging.getLogger('timeseries_training')
    
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config['learning_rate'],
        hidden_size=config['hidden_size'],
        attention_head_size=config['attention_head_size'],
        dropout=config['dropout'],
        hidden_continuous_size=config['hidden_continuous_size'],
        lstm_layers=config['lstm_layers'],
        output_size=7,  # Quantile output
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


# =============================================================================
# Training
# =============================================================================

def train_model(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    config: Dict,
    logger_inst: logging.Logger
) -> TemporalFusionTransformer:
    """Train the TFT model."""
    
    # Create dataloaders
    train_dataloader = DataLoader(
        training_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = create_tft_model(training_dataset, config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint_dir'],
        filename='tft-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config['early_stopping_patience'],
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Loggers
    tb_logger = TensorBoardLogger(
        save_dir=config['log_dir'],
        name='tft_training'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu',
        devices=config['gpus'],
        strategy=config['strategy'] if torch.cuda.device_count() > 1 else 'auto',
        gradient_clip_val=config['gradient_clip_val'],
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            RichProgressBar()
        ],
        logger=tb_logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    logger_inst.info("Starting training...")
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    logger_inst.info(f"Best model: {best_model_path}")
    
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    return best_model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(
    model: TemporalFusionTransformer,
    validation_dataset: TimeSeriesDataSet,
    config: Dict
) -> Dict[str, float]:
    """Evaluate the trained model."""
    logger = logging.getLogger('timeseries_training')
    
    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Predictions
    predictions = model.predict(val_dataloader)
    
    # Calculate metrics
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    
    mae = torch.mean(torch.abs(predictions - actuals)).item()
    rmse = torch.sqrt(torch.mean((predictions - actuals) ** 2)).item()
    mape = torch.mean(torch.abs((predictions - actuals) / actuals)).item() * 100
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }
    
    logger.info(f"Validation Metrics: {metrics}")
    
    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Time Series Forecasting Model')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--data', type=str, help='Path to data CSV')
    parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--gpus', type=int, default=-1, help='Number of GPUs')
    
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Override with CLI args
    if args.data:
        config['data_path'] = args.data
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    if args.epochs:
        config['max_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.gpus >= 0:
        config['gpus'] = args.gpus
    
    # Setup
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    logger = setup_logging(config['log_dir'])
    logger.info("=" * 60)
    logger.info("Time Series Forecasting Training")
    logger.info("=" * 60)
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Log GPU info
    if torch.cuda.is_available():
        logger.info(f"GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load data
    logger.info("Loading data...")
    df = load_sensor_data(config['data_path'])
    
    # Preprocess
    logger.info("Preprocessing data...")
    training_dataset, validation_dataset = preprocess_data(df, config)
    
    # Train
    logger.info("Training model...")
    model = train_model(training_dataset, validation_dataset, config, logger)
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, validation_dataset, config)
    
    # Save metrics
    metrics_path = os.path.join(config['checkpoint_dir'], 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save model info
    model_info = {
        'model_type': 'TemporalFusionTransformer',
        'config': config,
        'metrics': metrics,
        'training_date': datetime.now().isoformat(),
    }
    
    info_path = os.path.join(config['checkpoint_dir'], 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
