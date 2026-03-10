#!/usr/bin/env python3
"""
Crop Recommendation Model Training Script
==========================================
Trains XGBoost and Neural Network models for crop recommendation based on
soil and environmental conditions.
Optimized for CINECA Leonardo HPC.

Author: GrowAI Team
Date: 2024
"""

import os
import sys
import json
import logging
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

# XGBoost
import xgboost as xgb

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Data
    'data_path': os.environ.get('SCRATCH', '.') + '/growai-hpc/data/crop_data.csv',
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
    
    # XGBoost
    'xgb_params': {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'device': 'cuda',
    },
    
    # Neural Network
    'nn_hidden_dims': [256, 128, 64],
    'nn_dropout': 0.3,
    'nn_batch_norm': True,
    'nn_learning_rate': 0.001,
    'nn_batch_size': 256,
    'nn_epochs': 100,
    'nn_early_stopping_patience': 15,
    
    # Training
    'cv_folds': 5,
    'use_xgb': True,
    'use_nn': True,
    'ensemble': True,
    
    # Output
    'checkpoint_dir': os.environ.get('WORK', '.') + '/growai-hpc/checkpoints/recommendation',
    'log_dir': os.environ.get('WORK', '.') + '/growai-hpc/logs/recommendation',
}

# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('recommendation_training')
    logger.setLevel(logging.INFO)
    
    log_file = os.path.join(log_dir, f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Data Loading & Preprocessing
# =============================================================================

def load_crop_data(data_path: str) -> pd.DataFrame:
    """Load and validate crop recommendation data."""
    logger = logging.getLogger('recommendation_training')
    
    if not os.path.exists(data_path):
        logger.warning(f"Data file not found: {data_path}")
        logger.info("Generating synthetic crop data for testing...")
        return generate_synthetic_crop_data()
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    return df


def generate_synthetic_crop_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic crop recommendation data."""
    np.random.seed(42)
    
    crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'vegetables', 'fruits', 'pulses']
    
    data = []
    for _ in range(n_samples):
        # Generate soil and environmental features
        n = np.random.uniform(0, 140)
        p = np.random.uniform(5, 145)
        k = np.random.uniform(5, 205)
        temperature = np.random.uniform(8, 43)
        humidity = np.random.uniform(14, 99)
        ph = np.random.uniform(3.5, 9.5)
        rainfall = np.random.uniform(20, 300)
        
        # Simple rule-based crop assignment
        if ph < 5.5:
            crop = np.random.choice(['rice', 'vegetables'])
        elif temperature > 30 and humidity > 70:
            crop = np.random.choice(['rice', 'sugarcane', 'cotton'])
        elif temperature < 20:
            crop = np.random.choice(['wheat', 'pulses'])
        elif rainfall > 200:
            crop = np.random.choice(['rice', 'sugarcane'])
        else:
            crop = np.random.choice(crops)
        
        data.append({
            'N': n,
            'P': p,
            'K': k,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall,
            'crop': crop
        })
    
    return pd.DataFrame(data)


def preprocess_data(df: pd.DataFrame, config: Dict) -> Tuple:
    """Preprocess data for training."""
    logger = logging.getLogger('recommendation_training')
    
    # Feature columns
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target_col = 'crop'
    
    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    logger.info(f"Classes: {label_encoder.classes_}")
    logger.info(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, 
        test_size=config['test_size'],
        random_state=config['random_state'],
        stratify=y_encoded
    )
    
    val_size_adjusted = config['val_size'] / (1 - config['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=config['random_state'],
        stratify=y_temp
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        scaler, label_encoder, feature_cols
    )


# =============================================================================
# Neural Network Model (PyTorch Lightning)
# =============================================================================

class CropRecommendationNN(pl.LightningModule):
    """Neural Network for crop recommendation."""
    
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        batch_norm: bool = True,
        learning_rate: float = 0.001
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    config: Dict,
    logger_inst: logging.Logger
) -> CropRecommendationNN:
    """Train neural network model."""
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['nn_batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['nn_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = CropRecommendationNN(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dims=config['nn_hidden_dims'],
        dropout=config['nn_dropout'],
        batch_norm=config['nn_batch_norm'],
        learning_rate=config['nn_learning_rate']
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint_dir'],
        filename='nn-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config['nn_early_stopping_patience'],
        mode='min'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['nn_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=-1 if torch.cuda.is_available() else 1,
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        logger=False
    )
    
    logger_inst.info("Training Neural Network...")
    trainer.fit(model, train_loader, val_loader)
    
    return model


# =============================================================================
# XGBoost Model
# =============================================================================

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict,
    logger_inst: logging.Logger
) -> xgb.XGBClassifier:
    """Train XGBoost model."""
    
    model = xgb.XGBClassifier(**config['xgb_params'])
    
    logger_inst.info("Training XGBoost...")
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    return model


# =============================================================================
# Ensemble & Evaluation
# =============================================================================

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str
) -> Dict[str, Any]:
    """Evaluate a model."""
    logger = logging.getLogger('recommendation_training')
    
    # Predictions
    if isinstance(model, CropRecommendationNN):
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            if torch.cuda.is_available():
                X_test_tensor = X_test_tensor.cuda()
                model = model.cuda()
            logits = model(X_test_tensor)
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    else:
        y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    logger.info(f"\n{model_name} Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'classification_report': classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_,
            output_dict=True
        ),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }


def create_ensemble(
    xgb_model: Optional[xgb.XGBClassifier],
    nn_model: Optional[CropRecommendationNN],
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict
) -> Dict[str, float]:
    """Create ensemble weights based on validation performance."""
    weights = {}
    
    if xgb_model is not None:
        xgb_pred = xgb_model.predict_proba(X_val)
        xgb_acc = accuracy_score(y_val, xgb_pred.argmax(axis=1))
        weights['xgb'] = xgb_acc
    
    if nn_model is not None:
        nn_model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val)
            nn_pred = torch.softmax(nn_model(X_val_tensor), dim=1).numpy()
        nn_acc = accuracy_score(y_val, nn_pred.argmax(axis=1))
        weights['nn'] = nn_acc
    
    # Normalize weights
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Crop Recommendation Model')
    parser.add_argument('--config', type=str, help='Path to config JSON')
    parser.add_argument('--data', type=str, help='Path to data CSV')
    parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, help='Number of epochs for NN')
    parser.add_argument('--no-xgb', action='store_true', help='Skip XGBoost')
    parser.add_argument('--no-nn', action='store_true', help='Skip Neural Network')
    
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Override CLI args
    if args.data:
        config['data_path'] = args.data
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    if args.epochs:
        config['nn_epochs'] = args.epochs
    if args.no_xgb:
        config['use_xgb'] = False
    if args.no_nn:
        config['use_nn'] = False
    
    # Setup
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    logger = setup_logging(config['log_dir'])
    logger.info("=" * 60)
    logger.info("Crop Recommendation Model Training")
    logger.info("=" * 60)
    
    # Load data
    logger.info("Loading data...")
    df = load_crop_data(config['data_path'])
    
    # Preprocess
    logger.info("Preprocessing data...")
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler, label_encoder, feature_cols
    ) = preprocess_data(df, config)
    
    num_classes = len(label_encoder.classes_)
    
    results = {}
    models = {}
    
    # Train XGBoost
    if config['use_xgb']:
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val, config, logger)
        models['xgb'] = xgb_model
        results['xgb'] = evaluate_model(
            xgb_model, X_test, y_test, label_encoder, 'XGBoost'
        )
        
        # Save XGBoost model
        xgb_path = os.path.join(config['checkpoint_dir'], 'xgb_model.json')
        xgb_model.save_model(xgb_path)
        logger.info(f"XGBoost model saved to {xgb_path}")
    
    # Train Neural Network
    if config['use_nn']:
        nn_model = train_neural_network(
            X_train, y_train, X_val, y_val, num_classes, config, logger
        )
        models['nn'] = nn_model
        results['nn'] = evaluate_model(
            nn_model, X_test, y_test, label_encoder, 'Neural Network'
        )
        
        # Save NN model
        nn_path = os.path.join(config['checkpoint_dir'], 'nn_model.ckpt')
        nn_model.to_torchscript(nn_path.replace('.ckpt', '.pt'), method='trace')
        logger.info(f"Neural Network model saved")
    
    # Ensemble
    if config['ensemble'] and len(models) > 1:
        logger.info("\nCreating ensemble...")
        ensemble_weights = create_ensemble(
            models.get('xgb'),
            models.get('nn'),
            X_val, y_val,
            config
        )
        logger.info(f"Ensemble weights: {ensemble_weights}")
        
        results['ensemble_weights'] = ensemble_weights
    
    # Save artifacts
    artifacts = {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_cols': feature_cols,
        'config': config,
        'results': results,
        'training_date': datetime.now().isoformat()
    }
    
    artifacts_path = os.path.join(config['checkpoint_dir'], 'artifacts.pkl')
    with open(artifacts_path, 'wb') as f:
        pickle.dump(artifacts, f)
    logger.info(f"Artifacts saved to {artifacts_path}")
    
    # Save metrics JSON
    metrics_path = os.path.join(config['checkpoint_dir'], 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
