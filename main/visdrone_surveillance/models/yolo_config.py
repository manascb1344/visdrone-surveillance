#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/model_config.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_config')

class YOLOConfig:
    """
    Class for managing YOLO model configurations and training parameters.
    
    This class handles:
    1. Loading and saving model configurations
    2. Merging configurations with defaults
    3. Validating configuration parameters
    """
    
    # Default configuration for YOLOv8 training
    DEFAULT_CONFIG = {
        # Dataset parameters
        "path": "datasets/VisDrone-YOLO",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        
        # Classes configuration - will be overridden by actual dataset
        "nc": 10,  # VisDrone has 10 classes
        "names": {},
        
        # Training parameters
        "epochs": 100,
        "batch_size": 16,
        "imgsz": 640,
        "device": 0,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        "workers": 8,  # maximum number of dataloader workers
        "patience": 50,  # early stopping patience (epochs without improvement)
        "save_period": 10,  # save checkpoint every x epochs
        "freeze": 10,  # freeze layers until this layer during training
        
        # Optimizer parameters
        "lr0": 0.01,  # initial learning rate
        "lrf": 0.01,  # final learning rate (lr0 * lrf)
        "momentum": 0.937,  # SGD momentum/Adam beta1
        "weight_decay": 0.0005,  # optimizer weight decay
        "warmup_epochs": 3.0,  # warmup epochs
        "warmup_momentum": 0.8,  # warmup initial momentum
        "warmup_bias_lr": 0.1,  # warmup initial bias lr
        
        # Loss weights
        "box": 7.5,  # box loss gain
        "cls": 0.5,  # cls loss gain
        "dfl": 1.5,  # dfl loss gain
        "nbs": 64,  # nominal batch size
        
        # Model configuration
        "model": {
            "type": "yolov8",
            "size": "m",  # n, s, m, l, x
            "pretrained": True,
        },
        
        # Augmentation parameters
        "augment": True,
        "mosaic": 1.0,  # mosaic augmentation (0-1)
        "degrees": 0.0,  # rotation augmentation (+/- deg)
        "translate": 0.1,  # translation augmentation (+/- fraction)
        "scale": 0.5,  # scale augmentation (+/- gain)
        "shear": 0.0,  # shear augmentation (+/- deg)
        "perspective": 0.0,  # perspective augmentation (+/- fraction)
        "flipud": 0.0,  # vertical flip augmentation probability
        "fliplr": 0.5,  # horizontal flip augmentation probability
        "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
        "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
        "hsv_v": 0.4,  # image HSV-Value augmentation (fraction)
        
        # Evaluation parameters
        "val_conf": 0.001,  # validation confidence threshold
        "val_iou": 0.7,  # NMS IoU threshold for validation
        
        # Checkpoint parameters
        "project": "output/models",  # save to project/name
        "name": "yolov8_visdrone",  # save to project/name
        "exist_ok": True,  # existing project/name ok, do not increment
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize YOLOConfig with optional configuration file.
        
        Args:
            config_path: Path to the YAML configuration file (default: None)
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dict containing the configuration
        """
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return self.config
        
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Merge with default config
            self._merge_configs(user_config)
            logger.info(f"Loaded configuration from {config_path}")
            
            return self.config
        
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return self.config
    
    def _merge_configs(self, user_config: Dict[str, Any]):
        """
        Merge user configuration with default configuration.
        
        Args:
            user_config: User-provided configuration
        """
        # Handle nested model configuration if present
        if "model" in user_config:
            if "model" not in self.config:
                self.config["model"] = {}
            
            for k, v in user_config["model"].items():
                self.config["model"][k] = v
            
            # Remove model from user_config to avoid double processing
            del user_config["model"]
        
        # Merge top-level parameters
        for k, v in user_config.items():
            if k != "model":  # Model already handled
                self.config[k] = v
    
    def save_config(self, output_path: str) -> bool:
        """
        Save the current configuration to a YAML file.
        
        Args:
            output_path: Path to save the configuration file
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Saved configuration to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the model-specific configuration parameters.
        
        Returns:
            Dict containing model configuration
        """
        return self.config.get("model", {})
    
    def get_training_args(self) -> Dict[str, Any]:
        """
        Get the training arguments for YOLOv8.
        
        Returns:
            Dict containing training arguments
        """
        # Extract training-specific parameters
        train_args = {
            "data": self.config.get("path", "datasets/VisDrone-YOLO"),
            "epochs": self.config.get("epochs", 100),
            "imgsz": self.config.get("imgsz", 640),
            "batch": self.config.get("batch_size", 16),
            "device": self.config.get("device", 0),
            "workers": self.config.get("workers", 8),
            "patience": self.config.get("patience", 50),
            "freeze": self.config.get("freeze", 10),
            "save_period": self.config.get("save_period", 10),
            "lr0": self.config.get("lr0", 0.01),
            "lrf": self.config.get("lrf", 0.01),
            "momentum": self.config.get("momentum", 0.937),
            "weight_decay": self.config.get("weight_decay", 0.0005),
            "warmup_epochs": self.config.get("warmup_epochs", 3.0),
            "warmup_momentum": self.config.get("warmup_momentum", 0.8),
            "warmup_bias_lr": self.config.get("warmup_bias_lr", 0.1),
            "box": self.config.get("box", 7.5),
            "cls": self.config.get("cls", 0.5),
            "dfl": self.config.get("dfl", 1.5),
            "nbs": self.config.get("nbs", 64),
            "val": True,
            "save": True,
            "exist_ok": self.config.get("exist_ok", True),
            "project": self.config.get("project", "output/models"),
            "name": self.config.get("name", "yolov8_visdrone"),
        }
        
        # Add augmentation parameters if augment is enabled
        if self.config.get("augment", True):
            augment_params = {
                "mosaic": self.config.get("mosaic", 1.0),
                "degrees": self.config.get("degrees", 0.0),
                "translate": self.config.get("translate", 0.1),
                "scale": self.config.get("scale", 0.5),
                "shear": self.config.get("shear", 0.0),
                "perspective": self.config.get("perspective", 0.0),
                "flipud": self.config.get("flipud", 0.0),
                "fliplr": self.config.get("fliplr", 0.5),
                "hsv_h": self.config.get("hsv_h", 0.015),
                "hsv_s": self.config.get("hsv_s", 0.7),
                "hsv_v": self.config.get("hsv_v", 0.4),
            }
            train_args.update(augment_params)
        
        return train_args
    
    def get_model_path(self) -> str:
        """
        Get the path to the pretrained model based on configuration.
        
        Returns:
            str: Path to the model
        """
        model_config = self.get_model_config()
        model_type = model_config.get("type", "yolov8")
        model_size = model_config.get("size", "m")
        
        return f"{model_type}{model_size}.pt"
    
    def update_config(self, new_params: Dict[str, Any]):
        """
        Update the configuration with new parameters.
        
        Args:
            new_params: Dictionary containing new parameter values
        """
        for k, v in new_params.items():
            if k == "model" and isinstance(v, dict):
                if "model" not in self.config:
                    self.config["model"] = {}
                for mk, mv in v.items():
                    self.config["model"][mk] = mv
            else:
                self.config[k] = v
        
        logger.info("Updated configuration parameters")
    
    def print_config(self):
        """Print the current configuration."""
        print(json.dumps(self.config, indent=2))


def create_config_from_file(config_path: str, output_path: Optional[str] = None) -> YOLOConfig:
    """
    Create a YOLOConfig object from a configuration file.
    
    Args:
        config_path: Path to the input configuration file
        output_path: Optional path to save the merged configuration
        
    Returns:
        YOLOConfig: Configuration object
    """
    config = YOLOConfig(config_path)
    
    if output_path:
        config.save_config(output_path)
    
    return config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Model Configuration Tool")
    parser.add_argument('--config', type=str, default='configs/visdrone.yaml',
                        help='Path to the input configuration file')
    parser.add_argument('--output', type=str, default='configs/visdrone_merged.yaml',
                        help='Path to save the merged configuration')
    parser.add_argument('--print', action='store_true',
                        help='Print the merged configuration')
    args = parser.parse_args()
    
    # Create configuration
    config = create_config_from_file(args.config, args.output)
    
    # Print configuration if requested
    if args.print:
        config.print_config() 