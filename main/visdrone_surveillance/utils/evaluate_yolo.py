#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8 Evaluation Script for VisDrone dataset

This script evaluates a trained YOLOv8 model on the VisDrone dataset with:
- Performance metrics (mAP, precision, recall)
- Class-wise performance analysis
- Confusion matrix visualization
- Error analysis
- Visualization of detection results
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np

# Add parent directory to path

try:
    import torch
    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from tqdm import tqdm
    import cv2
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please run setup script first: python -m visdrone_surveillance.models.yolo_setup")
    sys.exit(1)

from ..models.yolo_config import YOLOConfig
from ..data import get_dataset_path, get_config_path

# Setup logging
def setup_logging(log_file=None):
    """Configure logging to file and console."""
    # Create logs directory if needed
    log_dir = Path('output/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # If no specific log file is provided, create a timestamped one
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'eval_{timestamp}.log'
    
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
    return logging.getLogger('yolo_eval')

def evaluate_model(
    model_path, 
    config_path=None, 
    output_dir=None, 
    conf_thres=0.25, 
    iou_thres=0.7, 
    save_json=True,
    save_hybrid=False,
    save_txt=False,
    visualize_predictions=False, 
    visualize_samples=10,
    split='val',
    device='0'  # Default device is GPU 0
):
    """
    Evaluate a YOLOv8 model on the specified dataset split.
    
    Args:
        model_path: Path to the YOLOv8 model weights (.pt file)
        config_path: Path to the configuration file (default: configs/visdrone_custom.yaml)
        output_dir: Directory to save evaluation results (default: output/eval/model_name)
        conf_thres: Confidence threshold for detections (default: 0.25)
        iou_thres: IoU threshold for NMS (default: 0.7)
        save_json: Save validation results to a JSON file (default: True)
        save_hybrid: Save hybrid version with labels (default: False)
        save_txt: Save detections as text files (default: False)
        visualize_predictions: Generate visualization of predictions (default: False)
        visualize_samples: Number of samples to visualize (default: 10)
        split: Dataset split to evaluate on ('val' or 'test')
        device: Device to run evaluation on ('0', '0,1', 'cpu', etc.)
        
    Returns:
        dict: Evaluation metrics
    """
    # Load model
    try:
        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
    
    # Load configuration
    if config_path is None:
        config_path = get_config_path('visdrone_custom.yaml')
    
    logger.info(f"Loading configuration from {config_path}")
    config = YOLOConfig()
    config.load_config(config_path)
    
    # Determine output directory
    if output_dir is None:
        model_name = Path(model_path).stem
        output_dir = Path('output/eval') / model_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Evaluation results will be saved to {output_dir}")
    
    # Set up evaluation arguments
    eval_args = {
        'data': str(config_path),
        'split': split,
        'conf': conf_thres,
        'iou': iou_thres,
        'project': str(output_dir.parent),
        'name': output_dir.name,
        'save_json': save_json,
        'save_hybrid': save_hybrid,
        'save_txt': save_txt,
        'device': device  # Use specified device
    }
    
    logger.info(f"Evaluating model with {eval_args}")
    
    # Start evaluation timer
    start_time = time.time()
    
    # Run evaluation
    try:
        results = model.val(**eval_args)
        
        # Extract metrics
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1-score': float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-16)),
        }
        
        # Calculate class-wise metrics if available
        if hasattr(results, 'classes') and results.classes is not None:
            class_names = config.config.get('names', {})
            class_metrics = {}
            
            for i, (ap50, ap) in enumerate(zip(results.ap50, results.ap)):
                class_name = class_names.get(str(i), f"class_{i}")
                class_metrics[class_name] = {
                    'mAP50': float(ap50),
                    'mAP50-95': float(ap),
                }
            
            metrics['class_metrics'] = class_metrics
        
        # Calculate timing metrics
        eval_time = time.time() - start_time
        metrics['eval_time'] = eval_time
        
        # Log and save results
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        logger.info(f"mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50-95']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        
        # Save metrics to JSON
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Visualize predictions if requested
        if visualize_predictions:
            visualize_detection_results(model, config_path, output_dir, split, visualize_samples)
        
        return metrics
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None

def visualize_detection_results(model, config_path, output_dir, split='val', num_samples=10):
    """
    Visualize detection results on a few samples.
    
    Args:
        model: Loaded YOLO model
        config_path: Path to the configuration file
        output_dir: Output directory for visualizations
        split: Dataset split to visualize ('val' or 'test')
        num_samples: Number of samples to visualize
    """
    # Load configuration
    config = YOLOConfig()
    config.load_config(config_path)
    
    # Get dataset path from config
    dataset_path = Path(config.config.get('path'))
    split_dir = dataset_path / split / 'images'
    
    if not split_dir.exists():
        logger.error(f"Split directory not found: {split_dir}")
        return
    
    # Create visualization directory
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Get image samples
    image_files = list(split_dir.glob('*.jpg'))
    if len(image_files) == 0:
        logger.error(f"No images found in {split_dir}")
        return
    
    # Select random samples if we have more than requested
    if len(image_files) > num_samples:
        import random
        random.shuffle(image_files)
        image_files = image_files[:num_samples]
    
    logger.info(f"Visualizing detections on {len(image_files)} images")
    
    # Get class names
    class_names = config.config.get('names', {})
    
    # Run inference and save visualizations
    for image_path in tqdm(image_files, desc="Generating visualizations"):
        # Run inference
        results = model.predict(str(image_path), conf=0.25, save=False)
        
        # Get result image with detections
        result_image = results[0].plot()
        
        # Save visualization
        output_path = vis_dir / f"{image_path.stem}_detection.jpg"
        cv2.imwrite(str(output_path), result_image)
    
    logger.info(f"Saved visualizations to {vis_dir}")

def plot_confusion_matrix(model, config_path, output_dir, split='val', device='0'):
    """
    Generate and plot confusion matrix for model predictions.
    
    Args:
        model: YOLO model 
        config_path: Path to configuration
        output_dir: Output directory
        split: Dataset split
        device: Device to run on ('0', 'cpu', etc.)
    """
    # Create output directory
    cm_dir = output_dir / 'confusion_matrix'
    cm_dir.mkdir(exist_ok=True)
    
    # Get class names
    config = YOLOConfig()
    config.load_config(config_path)
    class_names = config.config.get('names', {})
    
    logger.info("Generating confusion matrix...")
    
    # Run validation with confusion matrix
    try:
        results = model.val(
            data=config_path,
            split=split,
            conf=0.25,
            iou=0.7,
            save_json=False,
            plots=True,
            device=device
        )
        
        # Copy confusion matrix to our output dir
        run_dir = Path(results.save_dir)
        cm_file = run_dir / 'confusion_matrix.png'
        
        if cm_file.exists():
            import shutil
            shutil.copy(cm_file, cm_dir / 'confusion_matrix.png')
            logger.info(f"Saved confusion matrix to {cm_dir / 'confusion_matrix.png'}")
        else:
            logger.warning(f"Confusion matrix not found at {cm_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate confusion matrix: {e}")

def plot_precision_recall_curve(metrics, output_dir):
    """
    Plot precision-recall curves from evaluation metrics.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    if 'class_metrics' not in metrics:
        logger.warning("Class metrics not available, skipping PR curve")
        return
    
    # Create output directory
    pr_dir = output_dir / 'precision_recall'
    pr_dir.mkdir(exist_ok=True)
    
    # Setup plot
    plt.figure(figsize=(12, 8))
    
    # Plot class-wise PR curves
    class_metrics = metrics['class_metrics']
    for class_name, class_metric in class_metrics.items():
        # In this simplified version, we don't have actual PR curves
        # We just plot points for each class
        plt.scatter(
            class_metric.get('precision', 0),
            class_metric.get('recall', 0),
            label=f"{class_name} (AP={class_metric.get('mAP50', 0):.2f})"
        )
    
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall by Class')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig(pr_dir / 'precision_recall.png')
    logger.info(f"Saved precision-recall plot to {pr_dir / 'precision_recall.png'}")

def analyze_error_types(model, config_path, output_dir, split='val', num_samples=100):
    """
    Analyze error types (false positives, false negatives) in model predictions.
    
    Args:
        model: YOLO model
        config_path: Path to configuration
        output_dir: Output directory
        split: Dataset split
        num_samples: Maximum number of samples to analyze
    """
    # This is a more complex analysis that would require parsing prediction and ground truth
    # For this implementation, we'll use a simplified approach logging error counts
    
    # Create output directory
    error_dir = output_dir / 'error_analysis'
    error_dir.mkdir(exist_ok=True)
    
    logger.info("Performing error analysis...")
    
    # Run validation with save_json to get detailed predictions
    try:
        results = model.val(
            data=config_path,
            split=split,
            conf=0.25,
            iou=0.7,
            save_json=True
        )
        
        # Get path to JSON results
        run_dir = Path(results.save_dir)
        json_file = list(run_dir.glob('*.json'))
        
        if not json_file:
            logger.warning("No JSON results found for error analysis")
            return
        
        # Basic error analysis - count total FP and FN
        fp_count = int(results.box.fp_sum)
        fn_count = int(results.box.fn_sum)
        
        # Log error counts
        logger.info(f"False Positives: {fp_count}")
        logger.info(f"False Negatives: {fn_count}")
        
        # Save error counts to file
        error_stats = {
            'false_positives': fp_count,
            'false_negatives': fn_count,
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
        }
        
        with open(error_dir / 'error_stats.json', 'w') as f:
            json.dump(error_stats, f, indent=2)
        
        logger.info(f"Saved error analysis to {error_dir / 'error_stats.json'}")
        
    except Exception as e:
        logger.error(f"Error analysis failed: {e}")

def evaluate_across_confidence_thresholds(model, config_path, output_dir, split='val'):
    """
    Evaluate model across different confidence thresholds to analyze trade-offs.
    
    Args:
        model: YOLO model
        config_path: Path to configuration
        output_dir: Output directory
        split: Dataset split
    """
    # Define confidence thresholds to evaluate
    conf_thresholds = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    
    # Create output directory
    thresh_dir = output_dir / 'confidence_analysis'
    thresh_dir.mkdir(exist_ok=True)
    
    # Store results for each threshold
    threshold_results = []
    
    logger.info("Evaluating model across confidence thresholds...")
    
    for conf in conf_thresholds:
        logger.info(f"Evaluating at confidence threshold {conf}")
        
        try:
            results = model.val(
                data=config_path,
                split=split,
                conf=conf,
                iou=0.7,
                save_json=False
            )
            
            # Store metrics
            threshold_results.append({
                'conf_threshold': conf,
                'mAP50': float(results.box.map50),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1-score': float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-16)),
            })
            
        except Exception as e:
            logger.error(f"Evaluation failed at threshold {conf}: {e}")
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame(threshold_results)
    
    # Plot metrics vs. confidence threshold
    plt.figure(figsize=(12, 8))
    
    plt.plot(df['conf_threshold'], df['precision'], 'b-', label='Precision')
    plt.plot(df['conf_threshold'], df['recall'], 'r-', label='Recall')
    plt.plot(df['conf_threshold'], df['f1-score'], 'g-', label='F1-Score')
    plt.plot(df['conf_threshold'], df['mAP50'], 'k-', label='mAP50')
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs. Confidence Threshold')
    plt.grid(True)
    plt.legend()
    
    # Save plot and data
    plt.savefig(thresh_dir / 'threshold_analysis.png')
    df.to_csv(thresh_dir / 'threshold_results.csv', index=False)
    
    logger.info(f"Saved threshold analysis to {thresh_dir}")
    
    return df

def run_full_evaluation(
    model_path, 
    config_path=None, 
    output_dir=None, 
    split='val',
    visualize=True, 
    analyze_errors=True,
    analyze_thresholds=True,
    device='0'  # Default to GPU 0
):
    """
    Run a comprehensive evaluation of the model.
    
    Args:
        model_path: Path to the YOLOv8 model weights
        config_path: Path to configuration file
        output_dir: Output directory for results
        split: Dataset split to evaluate on
        visualize: Whether to visualize predictions
        analyze_errors: Whether to perform error analysis
        analyze_thresholds: Whether to analyze confidence thresholds
        device: Device to run on ('0', 'cpu', etc.)
        
    Returns:
        dict: Evaluation metrics
    """
    # Load model
    try:
        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
    
    # Determine configuration path
    if config_path is None:
        config_path = get_config_path('visdrone_custom.yaml')
    
    # Determine output directory
    if output_dir is None:
        model_name = Path(model_path).stem
        output_dir = Path('output/eval') / model_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Evaluation results will be saved to {output_dir}")
    
    # Run basic evaluation
    metrics = evaluate_model(
        model_path, 
        config_path, 
        output_dir, 
        visualize_predictions=visualize,
        split=split,
        device=device
    )
    
    if metrics is None:
        logger.error("Basic evaluation failed, stopping")
        return None
    
    # Plot confusion matrix
    try:
        plot_confusion_matrix(model, config_path, output_dir, split, device)
    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")
    
    # Plot precision-recall curve
    try:
        plot_precision_recall_curve(metrics, output_dir)
    except Exception as e:
        logger.error(f"Failed to plot precision-recall curve: {e}")
    
    # Analyze errors
    if analyze_errors:
        try:
            analyze_error_types(model, config_path, output_dir, split)
        except Exception as e:
            logger.error(f"Failed to analyze errors: {e}")
    
    # Evaluate across confidence thresholds
    if analyze_thresholds:
        try:
            threshold_df = evaluate_across_confidence_thresholds(model, config_path, output_dir, split)
            logger.info("Completed confidence threshold analysis")
        except Exception as e:
            logger.error(f"Failed to analyze confidence thresholds: {e}")
    
    logger.info("Comprehensive evaluation completed")
    return metrics

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 model on VisDrone dataset')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the YOLOv8 model weights (.pt file)')
    parser.add_argument('--config', type=str, default='configs/visdrone_custom.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--output', type=str, default=None,
                        help='Directory to save evaluation results')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='IoU threshold for NMS')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of predictions')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--comprehensive', action='store_true',
                        help='Run comprehensive evaluation (slower but more detailed)')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to log file')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to run on (e.g., "0" for GPU 0, "cpu" for CPU)')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log)
    logger.info("Starting YOLOv8 evaluation script")
    
    try:
        if args.comprehensive:
            # Run comprehensive evaluation
            metrics = run_full_evaluation(
                args.model,
                args.config,
                args.output,
                args.split,
                visualize=args.visualize,
                analyze_errors=True,
                analyze_thresholds=True,
                device=args.device
            )
        else:
            # Run basic evaluation
            metrics = evaluate_model(
                args.model,
                args.config,
                args.output,
                args.conf,
                args.iou,
                visualize_predictions=args.visualize,
                visualize_samples=args.samples,
                split=args.split,
                device=args.device
            )
        
        if metrics:
            logger.info("Evaluation completed successfully")
            for key, value in metrics.items():
                if key != 'class_metrics':  # Skip detailed class metrics
                    logger.info(f"{key}: {value}")
        else:
            logger.error("Evaluation failed")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Evaluation failed with error: {e}")
        sys.exit(1) 