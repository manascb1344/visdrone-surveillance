#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

from ultralytics import YOLO

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from project
from src.models.config import YOLOConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/model_factory.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_factory')

class YOLOModelFactory:
    """
    Factory class for creating and configuring YOLO models.
    
    This class handles:
    1. Creating YOLO models based on configuration
    2. Initializing with pretrained weights or from scratch
    3. Configuring model for training
    """
    
    def __init__(self, config: Optional[YOLOConfig] = None, config_path: Optional[str] = None):
        """
        Initialize YOLOModelFactory with configuration.
        
        Args:
            config: YOLOConfig object (default: None)
            config_path: Path to configuration file (ignored if config is provided)
        """
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = YOLOConfig(config_path)
        else:
            self.config = YOLOConfig()
        
        logger.info("Initialized YOLOModelFactory")
    
    def create_model(self) -> 'YOLO':
        """
        Create a YOLO model based on the configuration.
        
        Returns:
            YOLO: A YOLOv8 model
        """
        from ultralytics import YOLO
        
        model_config = self.config.get_model_config()
        model_type = model_config.get("type", "yolov8")
        model_size = model_config.get("size", "m")
        pretrained = model_config.get("pretrained", True)
        
        model_path = f"{model_type}{model_size}"
        model_file = f"{model_path}.pt"  # For pretrained
        model_yaml = f"{model_path}.yaml"  # For training from scratch
        
        logger.info(f"Creating {model_type}{model_size} model (pretrained={pretrained})")
        
        try:
            if pretrained:
                # Try to load pretrained model
                if os.path.exists(model_file):
                    logger.info(f"Loading pretrained model from {model_file}")
                    model = YOLO(model_file)
                else:
                    logger.info(f"Downloading pretrained model {model_file}")
                    model = YOLO(model_file)  # This will download the model
            else:
                # Train from scratch using YAML configuration
                logger.info(f"Creating model from scratch using {model_yaml}")
                model = YOLO(model_yaml)
            
            return model
        
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise
    
    def configure_model_for_training(
        self, 
        model: 'YOLO', 
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Configure a YOLO model for training with the current configuration.
        
        Args:
            model: YOLO model to configure
            experiment_name: Optional name for this training run
            
        Returns:
            Dict: Training arguments
        """
        # Get base training arguments
        train_args = self.config.get_training_args()
        
        # Override experiment name if provided
        if experiment_name:
            train_args["name"] = experiment_name
        
        logger.info(f"Configured model for training with {len(train_args)} parameters")
        logger.debug(f"Training configuration: {train_args}")
        
        return train_args
    
    def train_model(
        self, 
        data_config_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        resume: Optional[str] = None
    ) -> Tuple['YOLO', str]:
        """
        Create, configure and train a YOLO model in one step.
        
        Args:
            data_config_path: Path to data configuration YAML (default: use config.path)
            experiment_name: Optional name for this training run
            resume: Optional path to resume training from a checkpoint
            
        Returns:
            Tuple[YOLO, str]: Trained model and path to best weights
        """
        # Create model
        if resume:
            from ultralytics import YOLO
            logger.info(f"Resuming training from checkpoint: {resume}")
            model = YOLO(resume)
        else:
            model = self.create_model()
        
        # Configure training arguments
        train_args = self.configure_model_for_training(model, experiment_name)
        
        # Override data config if provided
        if data_config_path:
            train_args["data"] = data_config_path
        
        # Resume training
        if resume:
            train_args["resume"] = True
        
        # Log training configuration
        logger.info(f"Starting training with configuration: {train_args}")
        
        try:
            # Train the model
            results = model.train(**train_args)
            best_path = model.trainer.best
            logger.info(f"Training completed successfully. Best model: {best_path}")
            
            # Return the trained model and its path
            return model, best_path
        
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise
    
    def evaluate_model(
        self, 
        model_path: str, 
        data_config: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on validation dataset.
        
        Args:
            model_path: Path to the model weights
            data_config: Path to data configuration (default: use from config)
            
        Returns:
            Dict: Evaluation metrics
        """
        from ultralytics import YOLO
        
        # Use data config from object if not provided
        if data_config is None:
            data_config = self.config.config.get("path", "datasets/VisDrone-YOLO")
        
        logger.info(f"Evaluating model: {model_path} on {data_config}")
        
        try:
            # Load the model
            model = YOLO(model_path)
            
            # Run validation
            results = model.val(data=data_config)
            
            # Compile metrics
            metrics = {
                "mAP50": results.box.map50,
                "mAP50-95": results.box.map,
                "precision": results.box.mp,
                "recall": results.box.mr
            }
            
            logger.info(f"Evaluation metrics: {metrics}")
            return metrics
        
        except Exception as e:
            logger.error(f"Evaluation failed with error: {str(e)}")
            raise


def create_model_from_config(config_path: str) -> Tuple[YOLOModelFactory, 'YOLO']:
    """
    Create a model factory and model from a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Tuple[YOLOModelFactory, YOLO]: Model factory and model
    """
    factory = YOLOModelFactory(config_path=config_path)
    model = factory.create_model()
    
    return factory, model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Model Factory Tool")
    parser.add_argument('--config', type=str, default='configs/visdrone.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--create', action='store_true',
                        help='Create a model from the configuration')
    parser.add_argument('--train', action='store_true',
                        help='Train the model after creation')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume training from a checkpoint')
    parser.add_argument('--experiment', type=str, default='',
                        help='Name for the training experiment')
    parser.add_argument('--evaluate', type=str, default='',
                        help='Evaluate a trained model (provide path to weights)')
    args = parser.parse_args()
    
    # Create factory
    factory = YOLOModelFactory(config_path=args.config)
    
    # Create model if requested
    if args.create or args.train:
        model = factory.create_model()
        print(f"Created model with task: {model.task}")
    
    # Train model if requested
    if args.train:
        exp_name = args.experiment if args.experiment else None
        resume_path = args.resume if args.resume else None
        model, best_path = factory.train_model(experiment_name=exp_name, resume=resume_path)
        print(f"Training completed. Best model saved at: {best_path}")
    
    # Evaluate model if requested
    if args.evaluate:
        metrics = factory.evaluate_model(args.evaluate)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}") 