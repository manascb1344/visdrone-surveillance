import os
import glob
import shutil
import random
from tqdm import tqdm
import yaml
import argparse
from ultralytics import YOLO

def convert_visdrone_to_yolo(visdrone_path="visdrone", output_path="datasets/VisDrone-YOLO"):
    """
    Convert VisDrone annotations to YOLO format
    
    VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    Object categories: 0: ignored regions, 1: pedestrian, 2: person, 3: bicycle, 4: car, 5: van,
                      6: truck, 7: tricycle, 8: awning-tricycle, 9: bus, 10: motor, 11: others
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    (normalized coordinates)
    """
    # Create output directories
    os.makedirs(f"{output_path}/images/train", exist_ok=True)
    os.makedirs(f"{output_path}/images/val", exist_ok=True)
    os.makedirs(f"{output_path}/labels/train", exist_ok=True)
    os.makedirs(f"{output_path}/labels/val", exist_ok=True)
    
    # Define VisDrone to YOLO class mapping
    # We'll map similar classes and ignore some
    visdrone_to_yolo = {
        0: None,  # ignored regions -> ignore
        1: 0,     # pedestrian -> person
        2: 0,     # person -> person
        3: 1,     # bicycle -> bicycle
        4: 2,     # car -> car
        5: 3,     # van -> van
        6: 4,     # truck -> truck
        7: 5,     # tricycle -> custom
        8: 5,     # awning-tricycle -> custom
        9: 6,     # bus -> bus
        10: 7,    # motor -> motorcycle
        11: None  # others -> ignore
    }
    
    # Define class names
    class_names = ['person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motorcycle']
    
    # Create dataset.yaml file
    dataset_yaml = {
        'path': os.path.abspath(output_path),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(f"{output_path}/dataset.yaml", 'w') as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)
    
    # Process training set
    print("Converting training set...")
    train_imgs = glob.glob(f"{visdrone_path}/VisDrone2019-DET-train/images/*.jpg")
    for img_path in tqdm(train_imgs):
        img_name = os.path.basename(img_path)
        img_id = os.path.splitext(img_name)[0]
        
        # Copy image to output directory
        shutil.copy(img_path, f"{output_path}/images/train/{img_name}")
        
        # Convert annotation
        ann_path = f"{visdrone_path}/VisDrone2019-DET-train/annotations/{img_id}.txt"
        out_path = f"{output_path}/labels/train/{img_id}.txt"
        
        convert_annotation(ann_path, out_path, visdrone_to_yolo)
    
    # Process validation set
    print("Converting validation set...")
    val_imgs = glob.glob(f"{visdrone_path}/VisDrone2019-DET-val/images/*.jpg")
    for img_path in tqdm(val_imgs):
        img_name = os.path.basename(img_path)
        img_id = os.path.splitext(img_name)[0]
        
        # Copy image to output directory
        shutil.copy(img_path, f"{output_path}/images/val/{img_name}")
        
        # Convert annotation
        ann_path = f"{visdrone_path}/VisDrone2019-DET-val/annotations/{img_id}.txt"
        out_path = f"{output_path}/labels/val/{img_id}.txt"
        
        convert_annotation(ann_path, out_path, visdrone_to_yolo)
    
    print(f"Conversion complete. Dataset stored in {output_path}")
    return f"{output_path}/dataset.yaml"

def convert_annotation(ann_path, out_path, class_mapping):
    """Convert a single annotation file from VisDrone to YOLO format"""
    try:
        with open(ann_path, 'r') as f:
            lines = f.readlines()
        
        with open(out_path, 'w') as f:
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                
                # Get bounding box coordinates and class
                x = int(parts[0])
                y = int(parts[1])
                width = int(parts[2])
                height = int(parts[3])
                cls = int(parts[5])
                
                # Skip if class is ignored or mapped to None
                if cls not in class_mapping or class_mapping[cls] is None:
                    continue
                
                # Get image dimensions from the first line metadata if available
                # In real implementation, you might need to read the actual image dimensions
                img_width = 1920  # This should be dynamically determined from the image
                img_height = 1080  # This should be dynamically determined from the image
                
                # Convert to YOLO format: <class> <x_center> <y_center> <width> <height>
                yolo_x = (x + width / 2) / img_width
                yolo_y = (y + height / 2) / img_height
                yolo_width = width / img_width
                yolo_height = height / img_height
                
                # Ensure values are within bounds
                yolo_x = min(max(yolo_x, 0.0), 1.0)
                yolo_y = min(max(yolo_y, 0.0), 1.0)
                yolo_width = min(max(yolo_width, 0.0), 1.0)
                yolo_height = min(max(yolo_height, 0.0), 1.0)
                
                # Write to output file
                yolo_class = class_mapping[cls]
                # Skip if the mapped class is out of bounds for our class names list
                if yolo_class < 0 or yolo_class > 7:  # We have 8 classes (0-7)
                    continue
                f.write(f"{yolo_class} {yolo_x:.6f} {yolo_y:.6f} {yolo_width:.6f} {yolo_height:.6f}\n")
    except Exception as e:
        print(f"Error processing {ann_path}: {e}")

def validate_dataset(dataset_yaml):
    """
    Validate the dataset structure to ensure it's ready for training
    
    Args:
        dataset_yaml: Path to the dataset YAML file
    """
    import yaml
    import os
    
    print(f"Validating dataset from {dataset_yaml}...")
    
    # Check if YAML file exists
    if not os.path.exists(dataset_yaml):
        print(f"ERROR: Dataset YAML file not found: {dataset_yaml}")
        return False
    
    # Load YAML file
    with open(dataset_yaml, 'r') as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"ERROR: Failed to parse YAML file: {e}")
            return False
    
    # Check required keys
    required_keys = ['path', 'train', 'val', 'nc', 'names']
    missing_keys = [key for key in required_keys if key not in cfg]
    if missing_keys:
        print(f"ERROR: Missing required keys in YAML file: {missing_keys}")
        return False
    
    # Check paths
    base_path = cfg['path']
    train_path = os.path.join(base_path, cfg['train'])
    val_path = os.path.join(base_path, cfg['val'])
    
    if not os.path.exists(train_path):
        print(f"ERROR: Training images path not found: {train_path}")
        return False
    
    if not os.path.exists(val_path):
        print(f"ERROR: Validation images path not found: {val_path}")
        return False
    
    # Check for label files
    train_labels_path = os.path.join(base_path, 'labels', 'train')
    val_labels_path = os.path.join(base_path, 'labels', 'val')
    
    if not os.path.exists(train_labels_path):
        print(f"ERROR: Training labels path not found: {train_labels_path}")
        return False
    
    if not os.path.exists(val_labels_path):
        print(f"ERROR: Validation labels path not found: {val_labels_path}")
        return False
    
    # Check for images and corresponding label files
    import glob
    
    train_images = glob.glob(os.path.join(train_path, '*.jpg'))
    val_images = glob.glob(os.path.join(val_path, '*.jpg'))
    
    if not train_images:
        print(f"ERROR: No training images found in {train_path}")
        return False
    
    if not val_images:
        print(f"ERROR: No validation images found in {val_path}")
        return False
    
    # Check a sample of training images have corresponding label files
    for img_path in train_images[:5]:
        img_name = os.path.basename(img_path)
        img_id = os.path.splitext(img_name)[0]
        label_path = os.path.join(train_labels_path, f"{img_id}.txt")
        
        if not os.path.exists(label_path):
            print(f"WARNING: No label file for training image: {img_name}")
        else:
            # Check the label file has valid contents
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    print(f"WARNING: Empty label file for training image: {img_name}")
                else:
                    # Check first line has valid format
                    parts = lines[0].strip().split()
                    if len(parts) != 5:
                        print(f"WARNING: Invalid label format in {label_path}, expected 5 values but got {len(parts)}")
                    else:
                        try:
                            cls = int(parts[0])
                            if cls >= cfg['nc']:
                                print(f"WARNING: Class index {cls} out of bounds (max allowed: {cfg['nc']-1}) in {label_path}")
                        except ValueError:
                            print(f"WARNING: Invalid class value in {label_path}: {parts[0]}")
    
    print("Dataset validation completed successfully.")
    return True

def train_yolo_on_visdrone(dataset_yaml, model_size="n", epochs=3, batch_size=8, img_size=640):
    """
    Train YOLOv8 on the converted VisDrone dataset
    
    Args:
        dataset_yaml: Path to the dataset YAML file
        model_size: YOLO model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size for training
    """
    # Validate dataset first
    if not validate_dataset(dataset_yaml):
        print("Dataset validation failed. Aborting training.")
        return None
    
    print(f"Training YOLOv8{model_size} on VisDrone dataset for {epochs} epochs...")
    
    # Load pre-trained YOLOv8 model
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Train the model
    try:
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=20,  # Early stopping patience
            save=True,  # Save checkpoints
            device='cpu',  # Use CPU to avoid CUDA errors
            project="visdrone_yolo_training",
            name=f"yolov8{model_size}_visdrone",
            exist_ok=True,
            pretrained=True,
            optimizer="Adam"
        )
        
        print("Training completed.")
        return results
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Train YOLO on VisDrone dataset")
    parser.add_argument("--convert", action="store_true", help="Convert VisDrone to YOLO format")
    parser.add_argument("--train", action="store_true", help="Train YOLO on VisDrone")
    parser.add_argument("--model", type=str, default="n", help="YOLO model size (n, s, m, l, x)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img_size", type=int, default=640, help="Image size")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
        
    if args.all or args.convert:
        dataset_yaml = convert_visdrone_to_yolo()
    else:
        dataset_yaml = "datasets/VisDrone-YOLO/dataset.yaml"
    
    if args.all or args.train:
        train_yolo_on_visdrone(dataset_yaml, args.model, args.epochs, args.batch, args.img_size)

if __name__ == "__main__":
    main()
