#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from project
from src.data.dataset_utils import (
    ensure_yolov8_environment,
    verify_dataset_structure,
    create_visdrone_yaml,
    check_visdrone_yolo_format
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/yolo_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('yolo_setup')

def setup_yolov8_environment(dataset_path, config_path, force_recreate=False):
    """
    Setup the YOLOv8 environment and verify the dataset
    
    Args:
        dataset_path: Path to the VisDrone dataset
        config_path: Path to save the configuration file
        force_recreate: Whether to force recreate the YAML file
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    # Ensure environment
    logger.info("Checking YOLOv8 environment...")
    env_ready = ensure_yolov8_environment()
    
    if not env_ready:
        logger.error("YOLOv8 environment is not properly set up")
        return False
    
    logger.info("YOLOv8 environment is properly set up")
    
    # Check dataset path
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return False
    
    # Verify dataset structure
    logger.info(f"Verifying dataset structure at {dataset_path}")
    valid_structure = verify_dataset_structure(dataset_path)
    
    if not valid_structure:
        logger.warning("Dataset structure has inconsistencies. Some annotations might be missing.")
    
    # Check if in YOLO format
    is_yolo_format = check_visdrone_yolo_format(dataset_path)
    
    if not is_yolo_format:
        logger.error(f"Dataset at {dataset_path} is not in YOLO format")
        return False
    
    logger.info("Dataset is in YOLO format")
    
    # Create YAML file if it doesn't exist or force_recreate is True
    config_path = Path(config_path)
    if not config_path.exists() or force_recreate:
        logger.info(f"Creating YAML configuration at {config_path}")
        create_visdrone_yaml(config_path, str(dataset_path))
    else:
        logger.info(f"YAML configuration already exists at {config_path}")
    
    logger.info("YOLOv8 setup complete")
    return True

def download_pretrained_models(models=None, output_dir='.'):
    """
    Download pretrained YOLOv8 models
    
    Args:
        models: List of models to download (e.g., ['yolov8n.pt', 'yolov8s.pt'])
        output_dir: Directory to save models
    """
    from ultralytics import YOLO
    
    if models is None:
        models = ['yolov8n.pt', 'yolov8m.pt']  # Default models to download
    
    for model_name in models:
        logger.info(f"Downloading {model_name}...")
        
        output_path = Path(output_dir) / model_name
        if output_path.exists():
            logger.info(f"Model {model_name} already exists at {output_path}")
            continue
        
        try:
            # This will download the model if it doesn't exist
            model = YOLO(model_name)
            logger.info(f"Successfully downloaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup YOLOv8 environment and verify dataset")
    parser.add_argument('--dataset', type=str, default='datasets/VisDrone-YOLO',
                        help='Path to the VisDrone dataset')
    parser.add_argument('--config', type=str, default='configs/visdrone.yaml',
                        help='Path to save the YAML configuration file')
    parser.add_argument('--force-recreate', action='store_true',
                        help='Force recreate the YAML file even if it exists')
    parser.add_argument('--download-models', action='store_true',
                        help='Download pretrained YOLOv8 models')
    parser.add_argument('--models', type=str, nargs='+', default=['yolov8n.pt', 'yolov8m.pt'],
                        help='List of models to download')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Setup environment
    success = setup_yolov8_environment(args.dataset, args.config, args.force_recreate)
    
    if not success:
        logger.error("Failed to setup YOLOv8 environment")
        sys.exit(1)
    
    # Download models if requested
    if args.download_models:
        logger.info(f"Downloading pretrained models: {args.models}")
        download_pretrained_models(args.models)
    
    logger.info("YOLOv8 setup completed successfully") 