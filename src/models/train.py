#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8 Training Script for VisDrone dataset

This script implements YOLOv8 model training on the VisDrone dataset with:
- Configurable hyperparameters through YAML files
- Detailed logging
- Model checkpoints
- Metrics tracking
- TensorBoard support
"""

import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from datetime import datetime
import shutil
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    import torch
    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    from tqdm import tqdm
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please run setup script first: python src/models/yolo_setup.py")
    sys.exit(1)

from src.models.config import YOLOConfig
from src.data.dataset_utils import verify_dataset_structure, check_visdrone_yolo_format

# Setup logging
def setup_logging(log_file=None):
    """Configure logging to file and console."""
    # Create logs directory if needed
    log_dir = Path('output/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # If no specific log file is provided, create a timestamped one
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'train_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging configured. Log file: {log_file}")
    return logging.getLogger('yolo_train')

# Model Training Function
def train_yolo_model(config_path, resume=False, epochs=None):
    """
    Train a YOLOv8 model with the specified configuration.
    
    Args:
        config_path: Path to the configuration YAML file
        resume: Whether to resume training from a checkpoint
        epochs: Override the number of epochs from config file
    
    Returns:
        Path to trained model
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = YOLOConfig()
    config.load_config(config_path)
    
    if epochs is not None:
        config.update_config({'epochs': epochs})
        logger.info(f"Overriding epochs to {epochs}")
    
    # Verify dataset
    dataset_path = Path(config.config.get('path'))
    logger.info(f"Verifying dataset at {dataset_path}")
    
    is_valid = verify_dataset_structure(dataset_path)
    
    if not is_valid:
        logger.error(f"Dataset verification failed at {dataset_path}")
        sys.exit(1)
    
    is_yolo_format = check_visdrone_yolo_format(dataset_path)
    if not is_yolo_format:
        logger.error(f"Dataset is not in YOLO format at {dataset_path}")
        sys.exit(1)
    
    # Create output directories
    project_dir = Path(config.config.get('project'))
    model_name = config.config.get('name')
    output_dir = project_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration to output directory
    config_output_path = output_dir / 'config.yaml'
    config.save_config(config_output_path)
    logger.info(f"Saved configuration to {config_output_path}")
    
    # Initialize model
    model_config = config.get_model_config()
    model_size = model_config.get('size', 'm')
    pretrained = model_config.get('pretrained', True)
    
    if pretrained:
        logger.info(f"Initializing YOLOv8{model_size} model with pretrained weights")
        model = YOLO(f'yolov8{model_size}.pt')
    else:
        logger.info(f"Initializing YOLOv8{model_size} model from scratch")
        model = YOLO(f'yolov8{model_size}.yaml')
    
    # Set up training arguments
    train_args = {
        'data': str(config_path),
        'epochs': config.config.get('epochs', 100),
        'imgsz': config.config.get('imgsz', 640),
        'batch': config.config.get('batch_size', 16),
        'device': config.config.get('device', 0),
        'workers': config.config.get('workers', 8),
        'patience': config.config.get('patience', 50),
        'lr0': config.config.get('lr0', 0.01),
        'lrf': config.config.get('lrf', 0.01),
        'momentum': config.config.get('momentum', 0.937),
        'weight_decay': config.config.get('weight_decay', 0.0005),
        'warmup_epochs': config.config.get('warmup_epochs', 3.0),
        'warmup_momentum': config.config.get('warmup_momentum', 0.8),
        'warmup_bias_lr': config.config.get('warmup_bias_lr', 0.1),
        'box': config.config.get('box', 7.5),
        'cls': config.config.get('cls', 0.5),
        'dfl': config.config.get('dfl', 1.5),
        'project': str(project_dir),
        'name': model_name,
        'exist_ok': config.config.get('exist_ok', True),
        'pretrained': pretrained,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'save': True,
        'save_period': config.config.get('save_period', 10),
        'resume': resume,
        # Augmentation parameters
        'augment': config.config.get('augment', True),
        'mosaic': config.config.get('mosaic', 1.0),
        'degrees': config.config.get('degrees', 0.0),
        'translate': config.config.get('translate', 0.2),
        'scale': config.config.get('scale', 0.9),
        'shear': config.config.get('shear', 0.0),
        'perspective': config.config.get('perspective', 0.0),
        'flipud': config.config.get('flipud', 0.0),
        'fliplr': config.config.get('fliplr', 0.5),
        'hsv_h': config.config.get('hsv_h', 0.015),
        'hsv_s': config.config.get('hsv_s', 0.7),
        'hsv_v': config.config.get('hsv_v', 0.4),
        # Val config
        'val': True,
        'conf': config.config.get('val_conf', 0.001),
        'iou': config.config.get('val_iou', 0.6),
    }
    
    # Log training arguments
    logger.info("Training with the following arguments:")
    logger.info(json.dumps(train_args, indent=2))
    
    # Start training timer
    start_time = time.time()
    logger.info("Starting training...")
    
    # Train the model
    try:
        results = model.train(**train_args)
        
        # Log results summary
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logger.info(f"Final mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        logger.info(f"Final mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
        
        # Get best model path
        best_model_path = output_dir / 'weights' / 'best.pt'
        if best_model_path.exists():
            logger.info(f"Best model saved at: {best_model_path}")
            # Copy best model to the output root with meaningful name
            model_out_path = output_dir / f'visdrone_yolov8{model_size}_best.pt'
            shutil.copy(best_model_path, model_out_path)
            logger.info(f"Copied best model to: {model_out_path}")
            return model_out_path
        else:
            last_model_path = output_dir / 'weights' / 'last.pt'
            if last_model_path.exists():
                logger.info(f"Last model saved at: {last_model_path}")
                return last_model_path
                
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    return None

# Visualize Training Results
def plot_results(results_file, output_dir=None):
    """
    Plot training results from the results CSV file.
    
    Args:
        results_file: Path to the results.csv file
        output_dir: Directory to save the plots
    """
    if not output_dir:
        output_dir = Path(results_file).parent
    
    if not Path(results_file).exists():
        logger.error(f"Results file not found: {results_file}")
        return
    
    logger.info(f"Plotting results from {results_file}")
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Read results
        results = pd.read_csv(results_file)
        
        # Create plots directory
        plots_dir = Path(output_dir) / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Set plot style
        sns.set(style="darkgrid")
        
        # Plot training and validation losses
        plt.figure(figsize=(12, 6))
        plt.plot(results['epoch'], results['train/box_loss'], label='Train Box Loss')
        plt.plot(results['epoch'], results['train/cls_loss'], label='Train Class Loss')
        plt.plot(results['epoch'], results['train/dfl_loss'], label='Train DFL Loss')
        plt.plot(results['epoch'], results['val/box_loss'], label='Val Box Loss', linestyle='--')
        plt.plot(results['epoch'], results['val/cls_loss'], label='Val Class Loss', linestyle='--')
        plt.plot(results['epoch'], results['val/dfl_loss'], label='Val DFL Loss', linestyle='--')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(plots_dir / 'losses.png')
        
        # Plot mAP metrics
        plt.figure(figsize=(12, 6))
        plt.plot(results['epoch'], results['metrics/mAP50(B)'], label='mAP50')
        plt.plot(results['epoch'], results['metrics/mAP50-95(B)'], label='mAP50-95')
        plt.title('Validation mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.savefig(plots_dir / 'map.png')
        
        # Plot precision and recall
        plt.figure(figsize=(12, 6))
        plt.plot(results['epoch'], results['metrics/precision(B)'], label='Precision')
        plt.plot(results['epoch'], results['metrics/recall(B)'], label='Recall')
        plt.title('Precision and Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(plots_dir / 'precision_recall.png')
        
        logger.info(f"Plots saved to {plots_dir}")
        
    except Exception as e:
        logger.error(f"Failed to plot results: {e}")

# Main execution
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train YOLOv8 model on VisDrone dataset')
    parser.add_argument('--config', type=str, default='configs/visdrone_custom.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from last checkpoint')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs from config')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to log file')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log)
    logger.info("Starting YOLOv8 training script")
    
    # Log CUDA information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA not available. Training will be on CPU and may be very slow.")
    
    try:
        # Train the model
        model_path = train_yolo_model(args.config, args.resume, args.epochs)
        
        if model_path and Path(model_path).exists():
            # Plot training results
            results_csv = Path(model_path).parent.parent / 'results.csv'
            if results_csv.exists():
                plot_results(results_csv)
            
            logger.info("Training completed successfully!")
            logger.info(f"Trained model saved at: {model_path}")
        else:
            logger.error("Training failed or model not saved")
    
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        sys.exit(1) 