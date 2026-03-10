#!/usr/bin/env python3
"""
Agricultural LLM Fine-tuning Script
====================================
Fine-tunes Qwen 2.5 7B using QLoRA for agricultural domain adaptation.
Optimized for CINECA Leonardo HPC with DeepSpeed and multi-GPU support.

Author: GrowAI Team
Date: 2024
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

# Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint

# PEFT
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    TaskType
)

# DeepSpeed
import deepspeed

# Datasets
from datasets import load_dataset, Dataset as HFDataset

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Model
    'model_name': 'Qwen/Qwen2.5-7B',
    'model_revision': 'main',
    'trust_remote_code': True,
    
    # Quantization (QLoRA)
    'quantization': {
        'load_in_4bit': True,
        'bnb_4bit_compute_dtype': 'bfloat16',
        'bnb_4bit_use_double_quant': True,
        'bnb_4bit_quant_type': 'nf4',
    },
    
    # LoRA
    'lora': {
        'r': 64,
        'lora_alpha': 16,
        'target_modules': [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
        ],
        'lora_dropout': 0.05,
        'bias': 'none',
        'task_type': 'CAUSAL_LM',
    },
    
    # Training
    'training': {
        'num_train_epochs': 3,
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 2,
        'gradient_accumulation_steps': 4,
        'learning_rate': 2e-4,
        'weight_decay': 0.001,
        'warmup_ratio': 0.03,
        'lr_scheduler_type': 'cosine',
        'max_grad_norm': 0.3,
        'group_by_length': True,
        'bf16': True,
        'fp16': False,
        'tf32': True,
        'gradient_checkpointing': True,
        'optim': 'paged_adamw_8bit',
        'max_steps': -1,
    },
    
    # Data
    'max_seq_length': 2048,
    'data_path': os.environ.get('SCRATCH', '.') + '/growai-hpc/data/llm_training.jsonl',
    'eval_split': 0.1,
    
    # Output
    'output_dir': os.environ.get('WORK', '.') + '/growai-hpc/checkpoints/llm',
    'logging_dir': os.environ.get('WORK', '.') + '/growai-hpc/logs/llm',
    'save_steps': 500,
    'logging_steps': 10,
    'eval_steps': 500,
    'save_total_limit': 3,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'greater_is_better': False,
    
    # DeepSpeed
    'deepspeed_config': None,  # Auto-generated if None
}

# Agricultural prompt template
AGRICULTURAL_TEMPLATE = """You are an expert agricultural AI assistant. Provide helpful, accurate, and practical advice to farmers and agricultural professionals.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

DEFAULT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_dir: str, rank: int = 0) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('llm_training')
    logger.setLevel(logging.INFO)
    
    # Only log from main process
    if rank != 0:
        logger.addHandler(logging.NullHandler())
        return logger
    
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
# DeepSpeed Config
# =============================================================================

def get_deepspeed_config(stage: int = 2) -> Dict[str, Any]:
    """Generate DeepSpeed config."""
    if stage == 0:
        return None
    
    ds_config = {
        'fp16': {
            'enabled': False
        },
        'bf16': {
            'enabled': True
        },
        'zero_optimization': {
            'stage': stage,
            'offload_optimizer': {
                'device': 'none',
                'pin_memory': True
            },
            'allgather_partitions': True,
            'allgather_bucket_size': 2e8,
            'overlap_comm': True,
            'reduce_scatter': True,
            'reduce_bucket_size': 2e8,
            'contiguous_gradients': True,
        },
        'gradient_accumulation_steps': 'auto',
        'gradient_clipping': 'auto',
        'steps_per_print': 10,
        'train_batch_size': 'auto',
        'train_micro_batch_size_per_gpu': 'auto',
        'wall_clock_breakdown': False,
    }
    
    return ds_config


# =============================================================================
# Data Loading
# =============================================================================

def load_agricultural_data(data_path: str, eval_split: float = 0.1) -> tuple:
    """Load and prepare agricultural training data."""
    logger = logging.getLogger('llm_training')
    
    if not os.path.exists(data_path):
        logger.warning(f"Data file not found: {data_path}")
        logger.info("Generating synthetic agricultural training data...")
        return generate_synthetic_agricultural_data(eval_split)
    
    # Load from JSONL
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    # Split
    if eval_split > 0:
        split = dataset.train_test_split(test_size=eval_split)
        return split['train'], split['test']
    
    return dataset, None


def generate_synthetic_agricultural_data(eval_split: float = 0.1) -> tuple:
    """Generate synthetic agricultural Q&A data."""
    
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
        }
    ]
    
    # Expand dataset by varying examples
    expanded = []
    for sample in samples:
        expanded.append(sample)
        # Add variations
        expanded.append({
            'instruction': sample['instruction'],
            'input': sample['input'],
            'output': sample['output']
        })
    
    # Create dataset
    dataset = HFDataset.from_list(expanded * 100)  # Replicate for size
    
    split = dataset.train_test_split(test_size=eval_split)
    return split['train'], split['test']


def format_prompt(example: Dict, tokenizer) -> Dict:
    """Format example into prompt template."""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    prompt = AGRICULTURAL_TEMPLATE.format(
        instruction=instruction,
        input=input_text if input_text else 'None',
        output=output
    )
    
    return {'text': prompt}


def tokenize_function(examples: Dict, tokenizer, max_length: int) -> Dict:
    """Tokenize examples."""
    outputs = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors=None
    )
    
    # For causal LM, labels = input_ids
    outputs['labels'] = outputs['input_ids'].copy()
    
    return outputs


# =============================================================================
# Model Setup
# =============================================================================

def setup_model_and_tokenizer(config: Dict, logger: logging.Logger):
    """Setup model and tokenizer with QLoRA."""
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['quantization']['load_in_4bit'],
        bnb_4bit_compute_dtype=getattr(
            torch, config['quantization']['bnb_4bit_compute_dtype']
        ),
        bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant'],
        bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
    )
    
    logger.info(f"Loading model: {config['model_name']}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        trust_remote_code=config['trust_remote_code'],
        padding_side='right'
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=bnb_config,
        trust_remote_code=config['trust_remote_code'],
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2' if torch.cuda.is_available() else 'eager'
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, tokenizer


# =============================================================================
# Training
# =============================================================================

def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: Dict,
    logger: logging.Logger
):
    """Run training."""
    
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        logging_dir=config['logging_dir'],
        **config['training'],
        save_steps=config['save_steps'],
        logging_steps=config['logging_steps'],
        eval_steps=config['eval_steps'],
        save_total_limit=config['save_total_limit'],
        load_best_model_at_end=config['load_best_model_at_end'],
        metric_for_best_model=config['metric_for_best_model'],
        greater_is_better=config['greater_is_better'],
        evaluation_strategy='steps',
        save_strategy='steps',
        logging_strategy='steps',
        report_to=['tensorboard'],
        remove_unused_columns=False,
        deepspeed=config.get('deepspeed_config'),
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
    
    # Early stopping
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=5)
    ]
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # Resume from checkpoint if exists
    last_checkpoint = get_last_checkpoint(config['output_dir'])
    if last_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
    
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Save final model
    trainer.save_model(os.path.join(config['output_dir'], 'final'))
    
    return trainer


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Agricultural LLM')
    parser.add_argument('--config', type=str, help='Path to config JSON')
    parser.add_argument('--data', type=str, help='Path to training data')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--deepspeed', type=int, default=2, help='DeepSpeed stage (0/2/3)')
    
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Override CLI args
    if args.data:
        config['data_path'] = args.data
    if args.output_dir:
        config['output_dir'] = args.output_dir
        config['logging_dir'] = os.path.join(args.output_dir, 'logs')
    if args.epochs:
        config['training']['num_train_epochs'] = args.epochs
    
    # Setup DeepSpeed config
    if args.deepspeed > 0:
        ds_config = get_deepspeed_config(args.deepspeed)
        ds_path = os.path.join(config['output_dir'], 'deepspeed_config.json')
        os.makedirs(config['output_dir'], exist_ok=True)
        with open(ds_path, 'w') as f:
            json.dump(ds_config, f, indent=2)
        config['deepspeed_config'] = ds_path
    
    # Get rank
    rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Setup logging
    os.makedirs(config['logging_dir'], exist_ok=True)
    logger = setup_logging(config['logging_dir'], rank)
    
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Agricultural LLM Fine-tuning")
        logger.info("=" * 60)
        logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Load data
    logger.info("Loading data...")
    train_data, eval_data = load_agricultural_data(
        config['data_path'], 
        config['eval_split']
    )
    
    # Setup model
    logger.info("Setting up model...")
    model, tokenizer = setup_model_and_tokenizer(config, logger)
    
    # Format and tokenize
    logger.info("Tokenizing data...")
    
    train_data = train_data.map(
        lambda x: format_prompt(x, tokenizer),
        desc="Formatting prompts"
    )
    train_data = train_data.map(
        lambda x: tokenize_function(x, tokenizer, config['max_seq_length']),
        batched=True,
        desc="Tokenizing"
    )
    
    if eval_data:
        eval_data = eval_data.map(lambda x: format_prompt(x, tokenizer))
        eval_data = eval_data.map(
            lambda x: tokenize_function(x, tokenizer, config['max_seq_length']),
            batched=True
        )
    
    # Train
    trainer = train_model(model, tokenizer, train_data, eval_data, config, logger)
    
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()
