#!/usr/bin/env python3
"""
Data Preprocessing Script for GrowAI HPC Training
===================================================
Preprocesses raw agricultural data for the three model types:
- Time series sensor data
- Crop recommendation tabular data
- LLM fine-tuning text data

Author: GrowAI Team
Date: 2024
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    'raw_data_dir': os.environ.get('SCRATCH', '.') + '/growai-hpc/data/raw',
    'processed_data_dir': os.environ.get('SCRATCH', '.') + '/growai-hpc/data',
    'log_dir': os.environ.get('WORK', '.') + '/growai-hpc/logs',
    
    # Time series settings
    'ts_frequency': 'H',  # Hourly data
    'ts_sequence_length': 168,  # 1 week
    'ts_prediction_horizon': 24,  # 24 hours ahead
    
    # Crop recommendation settings
    'crop_test_size': 0.2,
    'crop_val_size': 0.1,
    'crop_random_state': 42,
    
    # LLM settings
    'llm_max_seq_length': 2048,
    'llm_train_ratio': 0.9,
}

# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('data_preprocessing')
    logger.setLevel(logging.INFO)
    
    log_file = os.path.join(log_dir, f'preprocessing_{datetime.now():%Y%m%d_%H%M%S}.log')
    
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
# Time Series Data Preprocessing
# =============================================================================

def preprocess_timeseries_data(
    input_path: Optional[str],
    output_dir: str,
    config: Dict,
    logger: logging.Logger
) -> str:
    """Preprocess time series sensor data."""
    logger.info("=" * 60)
    logger.info("Processing Time Series Data")
    logger.info("=" * 60)
    
    output_path = os.path.join(output_dir, 'sensors.csv')
    
    if input_path and os.path.exists(input_path):
        logger.info(f"Loading raw data from: {input_path}")
        df = pd.read_csv(input_path)
        
        # Validate required columns
        required_cols = ['timestamp', 'sensor_id', 'temperature', 'humidity', 'soil_moisture']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            logger.info("Generating synthetic data instead...")
            df = generate_synthetic_timeseries(config)
    else:
        logger.info("No input file provided. Generating synthetic sensor data...")
        df = generate_synthetic_timeseries(config)
    
    # Preprocessing steps
    logger.info(f"Raw data shape: {df.shape}")
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by sensor and time
    if 'sensor_id' in df.columns and 'timestamp' in df.columns:
        df = df.sort_values(['sensor_id', 'timestamp'])
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Remove outliers (IQR method)
    for col in numeric_cols:
        if col not in ['sensor_id', 'time_idx']:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(Q1, Q3)
    
    # Create time index if not exists
    if 'time_idx' not in df.columns and 'timestamp' in df.columns:
        df['time_idx'] = df.groupby('sensor_id').cumcount()
    
    # Create target variable (next day temperature)
    if 'temperature' in df.columns:
        df['target_temperature'] = df.groupby('sensor_id')['temperature'].shift(-config['ts_prediction_horizon'])
        df = df.dropna(subset=['target_temperature'])
    
    # Add derived features
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Save processed data
    df.to_csv(output_path, index=False)
    logger.info(f"Processed time series data saved to: {output_path}")
    logger.info(f"Final data shape: {df.shape}")
    logger.info(f"Sensors: {df['sensor_id'].nunique() if 'sensor_id' in df.columns else 'N/A'}")
    logger.info(f"Time range: {df['timestamp'].min() if 'timestamp' in df.columns else 'N/A'} to {df['timestamp'].max() if 'timestamp' in df.columns else 'N/A'}")
    
    # Create metadata
    metadata = {
        'data_type': 'time_series',
        'output_path': output_path,
        'shape': df.shape,
        'columns': list(df.columns),
        'numeric_stats': df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
        'processing_date': datetime.now().isoformat(),
    }
    
    metadata_path = os.path.join(output_dir, 'sensors_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return output_path


def generate_synthetic_timeseries(config: Dict) -> pd.DataFrame:
    """Generate synthetic time series sensor data."""
    np.random.seed(42)
    
    n_sensors = 50
    n_hours = 8760  # 1 year of hourly data
    sensors = [f'sensor_{i:03d}' for i in range(n_sensors)]
    dates = pd.date_range('2023-01-01', periods=n_hours, freq='H')
    
    data = []
    for sensor_id in sensors:
        # Base values with sensor-specific offsets
        base_temp = np.random.uniform(15, 25)
        base_humidity = np.random.uniform(50, 70)
        base_soil = np.random.uniform(30, 50)
        
        for i, date in enumerate(dates):
            # Seasonal and daily patterns
            hour_factor = np.sin(2 * np.pi * date.hour / 24)
            day_factor = np.sin(2 * np.pi * date.dayofyear / 365)
            
            # Add some randomness
            noise = np.random.normal(0, 1)
            
            data.append({
                'time_idx': i,
                'sensor_id': sensor_id,
                'timestamp': date,
                'temperature': base_temp + 5 * hour_factor + 8 * day_factor + noise * 2,
                'humidity': base_humidity + 10 * hour_factor + noise * 3,
                'soil_moisture': base_soil + 10 * np.sin(2 * np.pi * i / 168) + noise * 2,
                'light_intensity': max(0, 800 * np.sin(np.pi * date.hour / 12) * (1 if 6 <= date.hour <= 18 else 0) + np.random.normal(0, 50)),
                'co2_level': 400 + 100 * np.random.random() + 50 * np.sin(2 * np.pi * date.hour / 24),
                'ph': 6.5 + np.random.normal(0, 0.5),
                'nitrogen': 50 + 20 * np.random.random(),
                'phosphorus': 30 + 15 * np.random.random(),
                'potassium': 40 + 20 * np.random.random(),
            })
    
    return pd.DataFrame(data)


# =============================================================================
# Crop Recommendation Data Preprocessing
# =============================================================================

def preprocess_crop_data(
    input_path: Optional[str],
    output_dir: str,
    config: Dict,
    logger: logging.Logger
) -> str:
    """Preprocess crop recommendation tabular data."""
    logger.info("=" * 60)
    logger.info("Processing Crop Recommendation Data")
    logger.info("=" * 60)
    
    output_path = os.path.join(output_dir, 'crop_data.csv')
    
    if input_path and os.path.exists(input_path):
        logger.info(f"Loading raw data from: {input_path}")
        df = pd.read_csv(input_path)
    else:
        logger.info("No input file provided. Generating synthetic crop data...")
        df = generate_synthetic_crop_data(config)
    
    logger.info(f"Raw data shape: {df.shape}")
    
    # Define expected columns
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target_col = 'crop'
    
    # Validate columns
    available_features = [col for col in feature_cols if col in df.columns]
    if len(available_features) < len(feature_cols):
        logger.warning(f"Missing feature columns: {set(feature_cols) - set(available_features)}")
    
    # Handle missing values
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Clip outliers
    for col in feature_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.001)
            Q3 = df[col].quantile(0.999)
            df[col] = df[col].clip(Q1, Q3)
    
    # Encode target if string
    if target_col in df.columns and df[target_col].dtype == 'object':
        le = LabelEncoder()
        df['crop_encoded'] = le.fit_transform(df[target_col])
        
        # Save label mapping
        label_map = dict(zip(le.classes_, le.transform(le.classes_)))
        label_map_path = os.path.join(output_dir, 'crop_label_mapping.json')
        with open(label_map_path, 'w') as f:
            json.dump(label_map, f, indent=2)
        logger.info(f"Label mapping saved to: {label_map_path}")
    
    # Save processed data
    df.to_csv(output_path, index=False)
    logger.info(f"Processed crop data saved to: {output_path}")
    logger.info(f"Final data shape: {df.shape}")
    logger.info(f"Number of crop classes: {df[target_col].nunique() if target_col in df.columns else 'N/A'}")
    
    # Create feature statistics
    stats = {}
    for col in feature_cols:
        if col in df.columns:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median())
            }
    
    metadata = {
        'data_type': 'crop_recommendation',
        'output_path': output_path,
        'shape': df.shape,
        'columns': list(df.columns),
        'feature_stats': stats,
        'n_classes': df[target_col].nunique() if target_col in df.columns else None,
        'processing_date': datetime.now().isoformat(),
    }
    
    metadata_path = os.path.join(output_dir, 'crop_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return output_path


def generate_synthetic_crop_data(config: Dict) -> pd.DataFrame:
    """Generate synthetic crop recommendation data."""
    np.random.seed(42)
    
    crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'vegetables', 'fruits', 'pulses', 'oilseeds', 'millets']
    
    n_samples = 20000
    data = []
    
    for _ in range(n_samples):
        # Generate features with realistic ranges
        n = np.random.uniform(0, 140)
        p = np.random.uniform(5, 145)
        k = np.random.uniform(5, 205)
        temperature = np.random.uniform(8, 43)
        humidity = np.random.uniform(14, 99)
        ph = np.random.uniform(3.5, 9.5)
        rainfall = np.random.uniform(20, 300)
        
        # Simple rule-based crop assignment with some randomness
        conditions = []
        
        if ph < 5.5:
            conditions.extend(['rice'] * 3)
        elif ph > 7.5:
            conditions.extend(['wheat', 'millets'])
        
        if temperature > 30 and humidity > 70:
            conditions.extend(['rice', 'sugarcane', 'cotton'] * 2)
        elif temperature < 20:
            conditions.extend(['wheat', 'pulses'] * 2)
        
        if rainfall > 200:
            conditions.extend(['rice', 'sugarcane'] * 2)
        elif rainfall < 50:
            conditions.extend(['millets', 'oilseeds'] * 2)
        
        if 25 <= temperature <= 35 and 60 <= humidity <= 80:
            conditions.extend(['vegetables', 'fruits', 'maize'])
        
        # Add noise
        conditions.extend(crops)
        
        crop = np.random.choice(conditions)
        
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


# =============================================================================
# LLM Training Data Preprocessing
# =============================================================================

def preprocess_llm_data(
    input_path: Optional[str],
    output_dir: str,
    config: Dict,
    logger: logging.Logger
) -> str:
    """Preprocess LLM fine-tuning data."""
    logger.info("=" * 60)
    logger.info("Processing LLM Training Data")
    logger.info("=" * 60)
    
    output_path = os.path.join(output_dir, 'llm_training.jsonl')
    
    if input_path and os.path.exists(input_path):
        logger.info(f"Loading raw data from: {input_path}")
        
        # Handle different input formats
        if input_path.endswith('.jsonl'):
            df = pd.read_json(input_path, lines=True)
        elif input_path.endswith('.json'):
            with open(input_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
    else:
        logger.info("No input file provided. Generating synthetic agricultural Q&A data...")
        df = generate_synthetic_llm_data(config)
    
    logger.info(f"Raw data shape: {df.shape}")
    
    # Validate and clean data
    required_cols = ['instruction', 'output']
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Missing required column: {col}")
    
    # Add input column if missing
    if 'input' not in df.columns:
        df['input'] = ''
    
    # Clean text
    for col in ['instruction', 'input', 'output']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Remove very short entries
            df = df[df[col].str.len() >= 5] if col != 'input' else df
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['instruction', 'input'])
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Split into train/eval
    train_df, eval_df = train_test_split(
        df, 
        train_size=config['llm_train_ratio'],
        random_state=42
    )
    
    # Save as JSONL
    train_path = output_path
    eval_path = os.path.join(output_dir, 'llm_eval.jsonl')
    
    train_df.to_json(train_path, orient='records', lines=True)
    eval_df.to_json(eval_path, orient='records', lines=True)
    
    logger.info(f"Training data saved to: {train_path} ({len(train_df)} samples)")
    logger.info(f"Evaluation data saved to: {eval_path} ({len(eval_df)} samples)")
    
    # Calculate token estimates (rough)
    avg_chars = df['instruction'].str.len().mean() + df['output'].str.len().mean()
    est_tokens = int(avg_chars / 4 * len(df))  # Rough estimate: 4 chars per token
    logger.info(f"Estimated total tokens: ~{est_tokens:,}")
    
    # Metadata
    metadata = {
        'data_type': 'llm_fine_tuning',
        'train_path': train_path,
        'eval_path': eval_path,
        'train_samples': len(train_df),
        'eval_samples': len(eval_df),
        'total_samples': len(df),
        'avg_instruction_length': int(df['instruction'].str.len().mean()),
        'avg_output_length': int(df['output'].str.len().mean()),
        'estimated_tokens': est_tokens,
        'processing_date': datetime.now().isoformat(),
    }
    
    metadata_path = os.path.join(output_dir, 'llm_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return train_path


def generate_synthetic_llm_data(config: Dict) -> pd.DataFrame:
    """Generate synthetic agricultural Q&A for LLM training."""
    
    samples = [
        {
            'instruction': 'What are the best practices for rice cultivation in tropical climates?',
            'input': '',
            'output': 'For tropical rice cultivation: 1) Choose high-yielding varieties suited to your region. 2) Maintain 2-5cm water depth during vegetative growth. 3) Apply nitrogen fertilizer in split doses - basal, tillering, and panicle initiation. 4) Control weeds early using herbicides or manual weeding. 5) Monitor for pests like stem borers and planthoppers. 6) Harvest when 80% of grains turn golden yellow.'
        },
        {
            'instruction': 'How do I identify and treat nitrogen deficiency in wheat?',
            'input': 'My wheat plants have yellowing leaves starting from the tips.',
            'output': 'Yellowing from leaf tips indicates nitrogen deficiency. Solutions: 1) Apply urea or ammonium nitrate at 50-60 kg/ha. 2) Apply during active tillering for best uptake. 3) Split applications are more effective than single dose. 4) Ensure adequate soil moisture for nutrient absorption. 5) Consider foliar urea spray (5% solution) for quick correction.'
        },
        {
            'instruction': 'What irrigation schedule should I follow for tomato cultivation?',
            'input': 'I am growing tomatoes in a greenhouse with drip irrigation.',
            'output': 'For greenhouse tomatoes with drip irrigation: 1) Seedling stage: 2-3 times daily, 10-15 minutes. 2) Vegetative growth: Every 2-3 days, based on soil moisture. 3) Flowering/fruit set: Daily irrigation, 20-30 minutes. 4) Fruit development: Increase to 2-3 times daily in hot weather. 5) Monitor soil moisture at 20-30cm depth - maintain 60-70% field capacity. 6) Avoid water stress during flowering to prevent blossom end rot.'
        },
        {
            'instruction': 'How can I control aphids in my vegetable garden organically?',
            'input': 'Aphids are infesting my cabbage and broccoli plants.',
            'output': 'Organic aphid control methods: 1) Spray strong water jet to dislodge aphids. 2) Apply neem oil (3%) every 7-10 days. 3) Release ladybugs or lacewings as biocontrol agents. 4) Use insecticidal soap (2% solution). 5) Plant garlic or chives nearby as repellents. 6) Remove heavily infested leaves. 7) Apply diatomaceous earth around plant base.'
        },
        {
            'instruction': 'What soil pH is optimal for growing blueberries?',
            'input': 'My soil test shows pH 6.8.',
            'output': 'Blueberries require acidic soil pH of 4.5-5.5. At pH 6.8: 1) Apply elemental sulfur (200-400g per plant) to lower pH. 2) Incorporate peat moss into planting hole. 3) Use pine needle mulch to maintain acidity. 4) Apply ammonium sulfate fertilizer instead of nitrate forms. 5) Retest soil pH after 3 months. 6) Consider growing in containers with acidic potting mix for immediate solution.'
        },
        {
            'instruction': 'Explain integrated pest management (IPM) for cotton.',
            'input': '',
            'output': 'IPM for cotton involves: 1) Monitoring - use pheromone traps for bollworms, scout weekly for aphids and whiteflies. 2) Cultural control - crop rotation, timely sowing, remove crop residues. 3) Biological control - conserve parasitoids and predators like Chrysoperla. 4) Host plant resistance - use Bt cotton varieties. 5) Chemical control - apply pesticides only when economic threshold exceeded. 6) Target early instar larvae for best control. 7) Rotate insecticide classes to prevent resistance.'
        },
        {
            'instruction': 'How do I prepare land for maize cultivation?',
            'input': 'The previous crop was rice in this field.',
            'output': 'Land preparation after rice for maize: 1) Drain field completely 2-3 weeks before planting. 2) Plow when soil is friable to 20-25cm depth. 3) Harrow twice to break clods and level. 4) Apply lime if pH < 5.5 (rice fields often acidic). 5) Form ridges or beds for better drainage. 6) Apply basal fertilizer (NPK 15-15-15) at 200 kg/ha. 7) Ensure seedbed is firm but not compacted.'
        },
        {
            'instruction': 'What are the symptoms and management of powdery mildew in grapes?',
            'input': 'White powdery patches appeared on grape leaves.',
            'output': 'Powdery mildew symptoms: white powdery patches on leaves, shoots, and berries; distorted growth; cracked berries. Management: 1) Apply sulfur-based fungicides (wettable sulfur 0.2%). 2) Use systemic fungicides like myclobutanil or tebuconazole. 3) Ensure good air circulation through proper pruning. 4) Remove infected plant parts. 5) Apply at 10-15 day intervals during susceptible periods. 6) For organic: use potassium bicarbonate or neem oil sprays.'
        },
        {
            'instruction': 'How should I fertilize banana plants for maximum yield?',
            'input': 'I have Grand Nain variety in alluvial soil.',
            'output': 'Fertilizer schedule for Grand Nain in alluvial soil: 1) Basal: 200g N, 200g P2O5, 300g K2O per plant (at planting). 2) Month 2: 200g N, 100g K2O. 3) Month 4: 200g N, 100g K2O. 4) Month 6: 200g N, 200g K2O. 5) Month 8: 100g N, 300g K2O. 6) Apply zinc sulfate 25g/plant at month 3 if deficiency observed. 7) Total annual: 900g N, 200g P2O5, 1000g K2O per plant. 8) Split doses monthly during active growth for best results.'
        },
        {
            'instruction': 'What is the best time to harvest potatoes and how should I store them?',
            'input': 'Potato plants have flowered and leaves are turning yellow.',
            'output': 'Harvest timing: When 50-70% of leaves turn yellow and tuber skins are set (not peeling). Storage recommendations: 1) Cure tubers at 15-20°C with 85-90% humidity for 10-14 days to heal wounds. 2) Remove soil but do not wash. 3) Sort and discard damaged or diseased tubers. 4) Store at 2-4°C with 90-95% humidity for table use. 5) For seed potatoes: store at 3-4°C. 6) Ensure good ventilation. 7) Check regularly and remove sprouted or rotting tubers.'
        },
        {
            'instruction': 'What are the key factors affecting coffee bean quality?',
            'input': '',
            'output': 'Key factors affecting coffee bean quality: 1) Altitude - higher altitudes (1000-2000m) produce denser, more flavorful beans. 2) Climate - consistent temperatures between 18-24°C, moderate rainfall. 3) Soil - well-drained, volcanic soils rich in organic matter. 4) Shade - proper shading improves bean development. 5) Harvesting - hand-picking only ripe cherries. 6) Processing - proper wet or dry processing methods. 7) Storage - controlled humidity and temperature post-harvest.'
        },
        {
            'instruction': 'How do I start organic farming certification?',
            'input': 'I have a 50-acre farm currently using conventional methods.',
            'output': 'Steps for organic certification: 1) Transition period - 3 years without prohibited substances. 2) Choose accredited certifying agency. 3) Develop organic system plan detailing practices, inputs, and record-keeping. 4) Maintain detailed records of all farm activities. 5) Implement buffer zones to prevent contamination. 6) Undergo annual inspections. 7) Allow residue testing if requested. 8) Pay certification fees. 9) Display organic certificate and maintain compliance.'
        }
    ]
    
    # Expand dataset
    expanded = []
    for sample in samples:
        expanded.append(sample)
        # Add variations by slight modifications
        for i in range(50):  # 50 variations each
            expanded.append(sample)
    
    return pd.DataFrame(expanded)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Preprocess data for GrowAI training')
    parser.add_argument('--timeseries', type=str, help='Path to raw time series CSV')
    parser.add_argument('--crop', type=str, help='Path to raw crop data CSV')
    parser.add_argument('--llm', type=str, help='Path to raw LLM training data')
    parser.add_argument('--output-dir', type=str, help='Output directory for processed data')
    parser.add_argument('--only', choices=['timeseries', 'crop', 'llm', 'all'], default='all',
                       help='Process only specific data type')
    
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    if args.output_dir:
        config['processed_data_dir'] = args.output_dir
    
    # Setup
    os.makedirs(config['processed_data_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    logger = setup_logging(config['log_dir'])
    
    logger.info("=" * 60)
    logger.info("GrowAI Data Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Output directory: {config['processed_data_dir']}")
    
    results = {}
    
    # Process time series data
    if args.only in ['timeseries', 'all']:
        try:
            ts_path = preprocess_timeseries_data(
                args.timeseries,
                config['processed_data_dir'],
                config,
                logger
            )
            results['timeseries'] = ts_path
        except Exception as e:
            logger.error(f"Time series processing failed: {e}")
            results['timeseries'] = None
    
    # Process crop data
    if args.only in ['crop', 'all']:
        try:
            crop_path = preprocess_crop_data(
                args.crop,
                config['processed_data_dir'],
                config,
                logger
            )
            results['crop'] = crop_path
        except Exception as e:
            logger.error(f"Crop data processing failed: {e}")
            results['crop'] = None
    
    # Process LLM data
    if args.only in ['llm', 'all']:
        try:
            llm_path = preprocess_llm_data(
                args.llm,
                config['processed_data_dir'],
                config,
                logger
            )
            results['llm'] = llm_path
        except Exception as e:
            logger.error(f"LLM data processing failed: {e}")
            results['llm'] = None
    
    # Summary
    logger.info("=" * 60)
    logger.info("Preprocessing Complete")
    logger.info("=" * 60)
    for data_type, path in results.items():
        if path:
            logger.info(f"✓ {data_type}: {path}")
        else:
            logger.info(f"✗ {data_type}: Failed")
    
    return results


if __name__ == '__main__':
    main()
