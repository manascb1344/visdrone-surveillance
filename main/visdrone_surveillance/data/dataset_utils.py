#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import random
from pathlib import Path
from collections import Counter, defaultdict
import json
import yaml
import logging
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('dataset_utils')

# VisDrone class names
VISDRONE_CLASSES = {
    0: 'pedestrian',
    1: 'people',
    2: 'bicycle',
    3: 'car',
    4: 'van',
    5: 'truck',
    6: 'tricycle',
    7: 'awning-tricycle',
    8: 'bus',
    9: 'motor',
    10: 'others'
}

def get_dataset_paths(base_dir='datasets'):
    """
    Get paths to VisDrone dataset directories.
    
    Args:
        base_dir (str): Base directory containing the VisDrone dataset
        
    Returns:
        dict: Dictionary of dataset paths
    """
    return {
        'train': {
            'images': os.path.join(base_dir, 'VisDrone2019-DET-train', 'images'),
            'annotations': os.path.join(base_dir, 'VisDrone2019-DET-train', 'annotations')
        },
        'val': {
            'images': os.path.join(base_dir, 'VisDrone2019-DET-val', 'images'),
            'annotations': os.path.join(base_dir, 'VisDrone2019-DET-val', 'annotations')
        },
        'test': {
            'images': os.path.join(base_dir, 'VisDrone2019-DET-test-dev', 'images'),
            'annotations': os.path.join(base_dir, 'VisDrone2019-DET-test-dev', 'annotations')
        }
    }

def analyze_dataset_statistics(dataset_paths):
    """
    Analyze the VisDrone dataset statistics.
    
    Args:
        dataset_paths (dict): Dictionary of dataset paths
        
    Returns:
        dict: Dataset statistics
    """
    stats = {
        'train': {'count': 0, 'classes': Counter(), 'image_sizes': [], 'objects_per_image': []},
        'val': {'count': 0, 'classes': Counter(), 'image_sizes': [], 'objects_per_image': []},
        'test': {'count': 0, 'classes': Counter(), 'image_sizes': [], 'objects_per_image': []}
    }
    
    for split in ['train', 'val', 'test']:
        # Skip test if no annotations available
        if not os.path.exists(dataset_paths[split]['annotations']):
            continue
            
        image_dir = dataset_paths[split]['images']
        annotation_dir = dataset_paths[split]['annotations']
        
        image_files = [os.path.basename(f) for f in glob.glob(os.path.join(image_dir, '*.jpg'))]
        annotation_files = [os.path.basename(f).replace('.txt', '.jpg') 
                           for f in glob.glob(os.path.join(annotation_dir, '*.txt'))]
        
        # Find matching files
        matching_files = set(image_files).intersection(set(annotation_files))
        stats[split]['count'] = len(matching_files)
        
        print(f"Processing {split} set ({len(matching_files)} images)...")
        for image_file in tqdm(list(matching_files), desc=f"Analyzing {split} set"):
            # Get image dimensions
            img_path = os.path.join(image_dir, image_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            height, width = img.shape[:2]
            stats[split]['image_sizes'].append((width, height))
            
            # Get annotations
            annotation_path = os.path.join(annotation_dir, image_file.replace('.jpg', '.txt'))
            try:
                with open(annotation_path, 'r') as f:
                    lines = f.readlines()
                
                objects_count = 0
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) < 5:
                        continue
                        
                    class_id = int(parts[5])
                    # Filter out ignored regions (-1) and other invalid objects
                    if class_id >= 0 and class_id < len(VISDRONE_CLASSES):
                        stats[split]['classes'][class_id] += 1
                        objects_count += 1
                
                stats[split]['objects_per_image'].append(objects_count)
            except Exception as e:
                print(f"Error processing {annotation_path}: {e}")
    
    return stats

def plot_dataset_statistics(stats):
    """
    Plot dataset statistics.
    
    Args:
        stats (dict): Dataset statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs('output/dataset_analysis', exist_ok=True)
    
    # Plot class distribution
    plt.figure(figsize=(15, 8))
    for i, split in enumerate(['train', 'val']):
        if stats[split]['count'] == 0:
            continue
            
        plt.subplot(1, 2, i+1)
        class_counts = stats[split]['classes']
        labels = [VISDRONE_CLASSES[class_id] for class_id in sorted(class_counts.keys())]
        counts = [class_counts[class_id] for class_id in sorted(class_counts.keys())]
        
        plt.bar(labels, counts)
        plt.title(f'{split.capitalize()} Set Class Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    
    plt.savefig('output/dataset_analysis/class_distribution.png')
    
    # Plot objects per image
    plt.figure(figsize=(15, 6))
    for i, split in enumerate(['train', 'val']):
        if stats[split]['count'] == 0:
            continue
            
        plt.subplot(1, 2, i+1)
        objects_per_image = stats[split]['objects_per_image']
        plt.hist(objects_per_image, bins=20)
        plt.title(f'{split.capitalize()} Set Objects per Image')
        plt.xlabel('Number of Objects')
        plt.ylabel('Number of Images')
    
    plt.tight_layout()
    plt.savefig('output/dataset_analysis/objects_per_image.png')
    
    # Save statistics as JSON for further analysis
    with open('output/dataset_analysis/dataset_stats.json', 'w') as f:
        # Convert Counter objects to dictionaries
        json_stats = {}
        for split, split_stats in stats.items():
            json_stats[split] = {
                'count': split_stats['count'],
                'classes': {k: v for k, v in split_stats['classes'].items()},
                'objects_per_image_avg': np.mean(split_stats['objects_per_image']) if split_stats['objects_per_image'] else 0,
                'objects_per_image_max': max(split_stats['objects_per_image']) if split_stats['objects_per_image'] else 0,
                'objects_per_image_min': min(split_stats['objects_per_image']) if split_stats['objects_per_image'] else 0,
            }
        json.dump(json_stats, f, indent=2)
    
    print(f"Dataset statistics saved to output/dataset_analysis/")

def convert_visdrone_to_yolo(dataset_paths, output_dir='datasets/VisDrone-YOLO'):
    """
    Convert VisDrone annotations to YOLO format.
    
    Args:
        dataset_paths (dict): Dictionary of dataset paths
        output_dir (str): Output directory for YOLO format data
        
    Returns:
        dict: Paths to the converted dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create yaml file for dataset
    dataset_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': VISDRONE_CLASSES
    }
    
    with open(os.path.join(output_dir, 'visdrone.yaml'), 'w') as f:
        yaml_str = 'path: ' + dataset_yaml['path'] + '\n'
        yaml_str += 'train: ' + dataset_yaml['train'] + '\n'
        yaml_str += 'val: ' + dataset_yaml['val'] + '\n'
        yaml_str += 'test: ' + dataset_yaml['test'] + '\n'
        yaml_str += 'names:\n'
        for class_id, class_name in sorted(VISDRONE_CLASSES.items()):
            yaml_str += f'  {class_id}: {class_name}\n'
        f.write(yaml_str)
    
    for split in ['train', 'val', 'test']:
        # Skip test if no annotations available
        if not os.path.exists(dataset_paths[split]['annotations']):
            continue
            
        # Create output directories
        split_images_dir = os.path.join(output_dir, split, 'images')
        split_labels_dir = os.path.join(output_dir, split, 'labels')
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)
        
        image_dir = dataset_paths[split]['images']
        annotation_dir = dataset_paths[split]['annotations']
        
        image_files = [os.path.basename(f) for f in glob.glob(os.path.join(image_dir, '*.jpg'))]
        annotation_files = [os.path.basename(f).replace('.txt', '.jpg') 
                           for f in glob.glob(os.path.join(annotation_dir, '*.txt'))]
        
        # Find matching files
        matching_files = list(set(image_files).intersection(set(annotation_files)))
        
        print(f"Converting {split} set ({len(matching_files)} images)...")
        for image_file in tqdm(matching_files, desc=f"Converting {split} set"):
            # Copy image
            shutil.copy(
                os.path.join(image_dir, image_file), 
                os.path.join(split_images_dir, image_file)
            )
            
            # Get image dimensions
            img_path = os.path.join(image_dir, image_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            height, width = img.shape[:2]
            
            # Convert annotation
            annotation_path = os.path.join(annotation_dir, image_file.replace('.jpg', '.txt'))
            yolo_annotation_path = os.path.join(split_labels_dir, image_file.replace('.jpg', '.txt'))
            
            try:
                with open(annotation_path, 'r') as f_in, open(yolo_annotation_path, 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split(',')
                        if len(parts) < 8:  # VisDrone format has at least 8 parts
                            continue
                        
                        # VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                        x, y, w, h, _, class_id, _, _ = map(float, parts[:8])
                        class_id = int(class_id)
                        
                        # Skip ignored regions (-1) and other invalid classes
                        if class_id < 0 or class_id >= len(VISDRONE_CLASSES):
                            continue
                        
                        # Convert to YOLO format: <class_id> <x_center> <y_center> <width> <height>
                        # YOLO format uses normalized coordinates (0-1)
                        x_center = (x + w / 2) / width
                        y_center = (y + h / 2) / height
                        norm_width = w / width
                        norm_height = h / height
                        
                        # Check for invalid values
                        if not all(0 <= val <= 1 for val in [x_center, y_center, norm_width, norm_height]):
                            continue
                            
                        f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
            except Exception as e:
                print(f"Error processing {annotation_path}: {e}")
    
    print(f"Dataset converted to YOLO format in {output_dir}")
    return output_dir

def verify_dataset_integrity(yolo_dataset_dir):
    """
    Verify the integrity of the YOLO dataset.
    
    Args:
        yolo_dataset_dir (str): Path to the YOLO dataset directory
        
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    issues = []
    
    for split in ['train', 'val', 'test']:
        split_images_dir = os.path.join(yolo_dataset_dir, split, 'images')
        split_labels_dir = os.path.join(yolo_dataset_dir, split, 'labels')
        
        if not os.path.exists(split_images_dir) or not os.path.exists(split_labels_dir):
            continue
        
        image_files = set([os.path.splitext(os.path.basename(f))[0] 
                          for f in glob.glob(os.path.join(split_images_dir, '*.jpg'))])
        label_files = set([os.path.splitext(os.path.basename(f))[0] 
                          for f in glob.glob(os.path.join(split_labels_dir, '*.txt'))])
        
        # Check for missing labels or images
        missing_labels = image_files - label_files
        missing_images = label_files - image_files
        
        if missing_labels:
            issues.append(f"{split} set: Missing {len(missing_labels)} labels")
            
        if missing_images:
            issues.append(f"{split} set: Missing {len(missing_images)} images")
            
        # Check content of a random sample of label files
        sample_size = min(50, len(label_files))
        sample_labels = random.sample(list(label_files), sample_size)
        
        for label_name in sample_labels:
            label_path = os.path.join(split_labels_dir, f"{label_name}.txt")
            try:
                with open(label_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            issues.append(f"Invalid format in {label_path}, line {line_num}")
                            break
                            
                        class_id, x, y, w, h = parts
                        try:
                            class_id = int(class_id)
                            x, y, w, h = map(float, [x, y, w, h])
                            
                            if class_id < 0 or class_id >= len(VISDRONE_CLASSES):
                                issues.append(f"Invalid class ID {class_id} in {label_path}, line {line_num}")
                                
                            if not all(0 <= val <= 1 for val in [x, y, w, h]):
                                issues.append(f"Coordinates out of range in {label_path}, line {line_num}")
                        except ValueError:
                            issues.append(f"Invalid numeric values in {label_path}, line {line_num}")
            except Exception as e:
                issues.append(f"Error reading {label_path}: {e}")
    
    if issues:
        print("Dataset integrity issues found:")
        for issue in issues[:10]:  # Show only first 10 issues
            print(f"- {issue}")
        if len(issues) > 10:
            print(f"... and {len(issues) - 10} more issues")
        return False
    else:
        print("Dataset integrity check passed!")
        return True

def create_dataset_txt_files(yolo_dataset_dir):
    """
    Create train.txt, val.txt and test.txt files containing image paths.
    These files are needed by some frameworks for training.
    
    Args:
        yolo_dataset_dir (str): Path to the YOLO dataset directory
    """
    for split in ['train', 'val', 'test']:
        split_images_dir = os.path.join(yolo_dataset_dir, split, 'images')
        
        if not os.path.exists(split_images_dir):
            continue
            
        image_paths = glob.glob(os.path.join(split_images_dir, '*.jpg'))
        
        with open(os.path.join(yolo_dataset_dir, f'{split}.txt'), 'w') as f:
            for image_path in image_paths:
                f.write(f"{os.path.abspath(image_path)}\n")
                
    print(f"Created dataset txt files in {yolo_dataset_dir}")

def create_visdrone_yaml(output_path, dataset_path="datasets/VisDrone-YOLO"):
    """
    Create a YAML configuration file for the VisDrone dataset in YOLO format.
    
    Args:
        output_path: Path to save the YAML file
        dataset_path: Path to the VisDrone dataset
    """
    # VisDrone class names
    class_names = {
        0: 'pedestrian',
        1: 'people',
        2: 'bicycle',
        3: 'car',
        4: 'van',
        5: 'truck',
        6: 'tricycle',
        7: 'awning-tricycle',
        8: 'bus',
        9: 'motor',
        10: 'others'  # Add "others" class if present in the dataset
    }
    
    # Create config dict
    config = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': class_names,
        'nc': len(class_names)
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write YAML file
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created VisDrone YAML configuration at {output_path}")
    return output_path

def verify_dataset_structure(dataset_path, output_path=None):
    """
    Verify the VisDrone dataset structure and create train/val txt files if needed.
    
    Args:
        dataset_path: Path to the VisDrone dataset
        output_path: Path to save the train/val txt files (default: dataset_path)
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path) if output_path else dataset_path
    
    # Check if dataset directory exists
    if not dataset_path.exists():
        logger.error(f"Dataset directory does not exist: {dataset_path}")
        return False
    
    # Check train, val, test directories
    required_dirs = ['images/train', 'images/val', 'images/test', 
                     'labels/train', 'labels/val', 'labels/test']
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not (dataset_path / dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        logger.error(f"Missing required directories: {missing_dirs}")
        return False
    
    # Count images and labels
    train_images = list((dataset_path / 'images/train').glob('*.jpg'))
    val_images = list((dataset_path / 'images/val').glob('*.jpg'))
    test_images = list((dataset_path / 'images/test').glob('*.jpg'))
    
    train_labels = list((dataset_path / 'labels/train').glob('*.txt'))
    val_labels = list((dataset_path / 'labels/val').glob('*.txt'))
    test_labels = list((dataset_path / 'labels/test').glob('*.txt'))
    
    logger.info(f"Found {len(train_images)} training images, {len(val_images)} validation images, {len(test_images)} test images")
    logger.info(f"Found {len(train_labels)} training labels, {len(val_labels)} validation labels, {len(test_labels)} test labels")
    
    # Check if image count matches label count
    mismatched = False
    if len(train_images) != len(train_labels):
        logger.warning(f"Train image count ({len(train_images)}) does not match label count ({len(train_labels)})")
        mismatched = True
    
    if len(val_images) != len(val_labels):
        logger.warning(f"Validation image count ({len(val_images)}) does not match label count ({len(val_labels)})")
        mismatched = True
    
    if len(test_images) != len(test_labels):
        logger.warning(f"Test image count ({len(test_images)}) does not match label count ({len(test_labels)})")
        mismatched = True
    
    # Create train/val/test txt files
    train_txt = output_path / 'train.txt'
    val_txt = output_path / 'val.txt'
    test_txt = output_path / 'test.txt'
    
    # Write image paths to txt files
    with open(train_txt, 'w') as f:
        for img_path in train_images:
            f.write(str(img_path.absolute()) + '\n')
    
    with open(val_txt, 'w') as f:
        for img_path in val_images:
            f.write(str(img_path.absolute()) + '\n')
    
    with open(test_txt, 'w') as f:
        for img_path in test_images:
            f.write(str(img_path.absolute()) + '\n')
    
    logger.info(f"Created train.txt with {len(train_images)} images")
    logger.info(f"Created val.txt with {len(val_images)} images")
    logger.info(f"Created test.txt with {len(test_images)} images")
    
    return not mismatched

def check_visdrone_yolo_format(dataset_path):
    """
    Check if the VisDrone dataset is already in YOLO format
    
    Args:
        dataset_path: Path to the VisDrone dataset
        
    Returns:
        bool: True if dataset is in YOLO format, False otherwise
    """
    dataset_path = Path(dataset_path)
    
    # Check if labels directory exists and has YOLO format txt files
    train_labels_dir = dataset_path / 'labels/train'
    if not train_labels_dir.exists():
        return False
    
    # Check a sample label file to confirm YOLO format
    label_files = list(train_labels_dir.glob('*.txt'))
    if not label_files:
        return False
    
    # Check content of first label file
    try:
        with open(label_files[0], 'r') as f:
            first_line = f.readline().strip()
            # YOLO format should start with class index followed by coordinates
            parts = first_line.split()
            if len(parts) == 5:  # class_id x_center y_center width height
                try:
                    class_id = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                    # Check if coordinates are normalized (0-1)
                    if all(0 <= c <= 1 for c in coords):
                        return True
                except ValueError:
                    return False
    except (IOError, IndexError):
        return False
    
    return False

def ensure_yolov8_environment():
    """
    Ensure the YOLOv8 environment is properly set up
    
    Returns:
        bool: True if environment is properly set up, False otherwise
    """
    try:
        # Try importing required libraries
        import torch
        import ultralytics
        from ultralytics import YOLO
        
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Ultralytics version: {ultralytics.__version__}")
        logger.info(f"CUDA available: {cuda_available}")
        logger.info(f"GPU devices: {device_count}")
        
        if cuda_available:
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                logger.info(f"GPU {i}: {device_name}")
        
        # Try loading a small YOLOv8 model
        try:
            model = YOLO("yolov8n.pt")
            logger.info("Successfully loaded YOLOv8n model")
        except Exception as e:
            logger.warning(f"Failed to load YOLOv8n model: {str(e)}")
            return False
        
        return True
    
    except ImportError as e:
        logger.error(f"Missing required dependencies: {str(e)}")
        return False

def main():
    """Main function to preprocess VisDrone dataset."""
    print("VisDrone Dataset Preprocessing")
    print("=============================")
    
    # Get dataset paths
    dataset_paths = get_dataset_paths()
    
    # Analyze dataset statistics
    print("\n1. Analyzing dataset statistics...")
    stats = analyze_dataset_statistics(dataset_paths)
    plot_dataset_statistics(stats)
    
    # Convert to YOLO format
    print("\n2. Converting VisDrone annotations to YOLO format...")
    yolo_dataset_dir = convert_visdrone_to_yolo(dataset_paths)
    
    # Verify dataset integrity
    print("\n3. Verifying dataset integrity...")
    verify_dataset_integrity(yolo_dataset_dir)
    
    # Create dataset txt files
    print("\n4. Creating dataset txt files...")
    create_dataset_txt_files(yolo_dataset_dir)
    
    print("\nDataset preprocessing completed!")

if __name__ == "__main__":
    # Example usage
    logger.info("Checking YOLOv8 environment...")
    env_ready = ensure_yolov8_environment()
    
    if env_ready:
        logger.info("YOLOv8 environment is properly set up")
    else:
        logger.error("YOLOv8 environment setup failed")
        
    # Check and prepare VisDrone dataset
    visdrone_path = "datasets/VisDrone-YOLO"
    
    if os.path.exists(visdrone_path):
        logger.info(f"Verifying VisDrone dataset structure at {visdrone_path}")
        is_valid = verify_dataset_structure(visdrone_path)
        
        if is_valid:
            logger.info("VisDrone dataset structure is valid")
        else:
            logger.warning("VisDrone dataset structure has inconsistencies")
        
        # Create YAML config file
        create_visdrone_yaml("configs/visdrone.yaml", visdrone_path)
    else:
        logger.error(f"VisDrone dataset not found at {visdrone_path}")

    main() 