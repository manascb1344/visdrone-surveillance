import os
import sys
import argparse
import requests
import zipfile
import shutil
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np

# VisDrone dataset URLs
VISDRONE_URLS = {
    'train': 'https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-train.zip',
    'val': 'https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-val.zip',
    'test-dev': 'https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-test-dev.zip',
    'test-challenge': 'https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-test-challenge.zip'
}

# VisDrone to YOLO class mapping
# VisDrone classes: ignored regions(0), pedestrian(1), people(2), bicycle(3), car(4),
#                  van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9),
#                  motor(10), others(11)
# We'll ignore class 0 (ignored regions) and class 11 (others)
VISDRONE_TO_YOLO = {
    1: 0,  # pedestrian
    2: 1,  # people
    3: 2,  # bicycle
    4: 3,  # car
    5: 4,  # van
    6: 5,  # truck
    7: 6,  # tricycle
    8: 7,  # awning-tricycle
    9: 8,  # bus
    10: 9  # motor
}

def download_file(url, dest_path):
    """Download a file from a URL to a destination path with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return dest_path

def extract_zip(zip_path, extract_dir):
    """Extract a zip file to a directory"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total number of files for progress bar
        total_files = len(zip_ref.infolist())
        
        # Extract with progress bar
        with tqdm(total=total_files, desc=f"Extracting {os.path.basename(zip_path)}") as pbar:
            for file in zip_ref.infolist():
                zip_ref.extract(file, extract_dir)
                pbar.update(1)
    
    return extract_dir

def visdrone_to_yolo(annotation_path, img_width, img_height):
    """
    Convert VisDrone annotation format to YOLO format
    
    VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    YOLO format: <class> <x_center> <y_center> <width> <height>
    """
    yolo_annotations = []
    
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            
            # Extract values
            x = int(parts[0])
            y = int(parts[1])
            width = int(parts[2])
            height = int(parts[3])
            category = int(parts[5])
            
            # Skip ignored regions and 'others' category
            if category == 0 or category == 11:
                continue
            
            # Check if the category is in our mapping
            if category not in VISDRONE_TO_YOLO:
                continue
            
            # Convert to YOLO format
            yolo_class = VISDRONE_TO_YOLO[category]
            x_center = (x + width / 2) / img_width
            y_center = (y + height / 2) / img_height
            yolo_width = width / img_width
            yolo_height = height / img_height
            
            # Ensure values are within [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            yolo_width = max(0, min(1, yolo_width))
            yolo_height = max(0, min(1, yolo_height))
            
            yolo_annotations.append(f"{yolo_class} {x_center} {y_center} {yolo_width} {yolo_height}")
    
    return yolo_annotations

def process_dataset(input_dir, output_dir, dataset_type):
    """
    Process VisDrone dataset to YOLO format
    
    Args:
        input_dir: Path to the extracted VisDrone dataset
        output_dir: Path to save the processed dataset
        dataset_type: Type of dataset (train, val, test-dev, test-challenge)
    """
    # Create output directories
    images_dir = os.path.join(output_dir, 'images', dataset_type)
    labels_dir = os.path.join(output_dir, 'labels', dataset_type)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Get annotations and images directories
    if dataset_type in ['train', 'val']:
        annotations_dir = os.path.join(input_dir, 'annotations')
        images_src_dir = os.path.join(input_dir, 'images')
    else:
        annotations_dir = os.path.join(input_dir, 'annotations')
        images_src_dir = os.path.join(input_dir, 'images')
    
    # Process all images
    image_files = sorted([f for f in os.listdir(images_src_dir) if f.endswith('.jpg')])
    
    image_paths = []  # Store image paths for the list file
    
    for img_file in tqdm(image_files, desc=f"Processing {dataset_type}"):
        # Image paths
        img_src_path = os.path.join(images_src_dir, img_file)
        img_dst_path = os.path.join(images_dir, img_file)
        
        # Get image dimensions
        img = cv2.imread(img_src_path)
        if img is None:
            print(f"Warning: Could not read image {img_src_path}")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Copy image
        shutil.copy2(img_src_path, img_dst_path)
        image_paths.append(os.path.join('images', dataset_type, img_file))
        
        # Process annotation if available
        ann_file = img_file.replace('.jpg', '.txt')
        ann_src_path = os.path.join(annotations_dir, ann_file)
        ann_dst_path = os.path.join(labels_dir, ann_file)
        
        if os.path.exists(ann_src_path):
            yolo_annotations = visdrone_to_yolo(ann_src_path, img_width, img_height)
            with open(ann_dst_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
    
    # Create list file
    list_file_path = os.path.join(output_dir, f"{dataset_type}.txt")
    with open(list_file_path, 'w') as f:
        f.write('\n'.join(image_paths))
    
    return len(image_paths)

def prepare_visdrone(data_dir, output_dir, download=True, datasets=None):
    """
    Download and preprocess VisDrone dataset
    
    Args:
        data_dir: Directory to download and extract the dataset
        output_dir: Directory to save the processed dataset
        download: Whether to download the dataset (or use existing files)
        datasets: List of dataset types to process (default: all)
    """
    if datasets is None:
        datasets = ['train', 'val', 'test-dev']
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Download and process each dataset
    for dataset in datasets:
        print(f"\nProcessing {dataset} dataset...")
        
        if download:
            # Download the dataset
            zip_path = os.path.join(data_dir, f"VisDrone2019-DET-{dataset}.zip")
            if not os.path.exists(zip_path):
                print(f"Downloading {dataset} dataset...")
                download_file(VISDRONE_URLS[dataset], zip_path)
            
            # Extract the dataset
            extract_dir = os.path.join(data_dir, f"VisDrone2019-DET-{dataset}")
            if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
                print(f"Extracting {dataset} dataset...")
                extract_zip(zip_path, data_dir)
        
        # Process the dataset
        input_dir = os.path.join(data_dir, f"VisDrone2019-DET-{dataset}")
        if not os.path.exists(input_dir):
            print(f"Error: {input_dir} does not exist. Please download the dataset first.")
            continue
        
        # Map test-dev and test-challenge to test for output
        output_type = 'test' if dataset in ['test-dev', 'test-challenge'] else dataset
        processed_count = process_dataset(input_dir, output_dir, output_type)
        print(f"Processed {processed_count} images for {dataset} dataset")
    
    # Create YAML configuration
    classes = {v: k for k, v in VISDRONE_TO_YOLO.items()}
    class_names = [
        'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    # Create a sample YAML file
    yaml_path = os.path.join(output_dir, 'visdrone.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write("test: test.txt\n\n")
        
        f.write("names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")
    
    print(f"\nDataset preprocessing complete. YAML configuration saved to {yaml_path}")
    print(f"Total images processed: {sum(1 for p in Path(output_dir).glob('images/**/*.jpg'))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess VisDrone dataset")
    parser.add_argument('--data-dir', type=str, default='datasets/VisDrone-raw',
                        help='Directory to download and extract the dataset')
    parser.add_argument('--output-dir', type=str, default='datasets/VisDrone-YOLO',
                        help='Directory to save the processed dataset')
    parser.add_argument('--no-download', action='store_true',
                        help='Skip downloading and use existing files')
    parser.add_argument('--datasets', type=str, nargs='+', choices=['train', 'val', 'test-dev', 'test-challenge'],
                        default=['train', 'val', 'test-dev'],
                        help='Dataset types to process')
    
    args = parser.parse_args()
    
    prepare_visdrone(args.data_dir, args.output_dir, not args.no_download, args.datasets) 