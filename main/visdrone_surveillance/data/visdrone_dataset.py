import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
from tqdm import tqdm
import sys

from .. import config

class VisDroneDataset(Dataset):
    """
    Dataset class for VisDrone object detection dataset
    """
    def __init__(self, sequences_dir, annotations_dir, transform=None, train=True, validation_split=0.2, seed=42):
        self.sequences_dir = sequences_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.train = train
        self.validation_split = validation_split
        self.seed = seed
        
        # Get all sequence folders
        self.sequence_folders = sorted(os.listdir(sequences_dir))
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Split data into train and validation
        indices = np.arange(len(self.sequence_folders))
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - validation_split))
        
        if train:
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
            
        # Extract image paths and their corresponding annotations
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """
        Load all image samples and their corresponding annotations
        """
        samples = []
        
        print(f"Loading {'training' if self.train else 'validation'} samples...")
        for idx in tqdm(self.indices):
            sequence_folder = self.sequence_folders[idx]
            
            # Get all image files in this sequence
            image_files = sorted(glob.glob(os.path.join(self.sequences_dir, sequence_folder, "*.jpg")))
            
            # Load annotation file for this sequence
            annotation_file = os.path.join(self.annotations_dir, f"{sequence_folder}.txt")
            
            if os.path.exists(annotation_file):
                # Group annotations by frame number
                frame_annotations = {}
                
                with open(annotation_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        frame_id = int(parts[0])
                        
                        if frame_id not in frame_annotations:
                            frame_annotations[frame_id] = []
                        
                        # Extract bbox and class
                        # Format: [frame_id, object_id, bbox_x, bbox_y, bbox_width, bbox_height, score, object_category, truncation, occlusion]
                        # We care about bbox and object_category
                        x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                        obj_category = int(parts[7])
                        
                        # Skip ignored regions (category 0)
                        if obj_category > 0:
                            frame_annotations[frame_id].append([x, y, w, h, obj_category])
                
                # Match images with their annotations
                for img_path in image_files:
                    img_name = os.path.basename(img_path)
                    frame_id = int(img_name.split('.')[0])
                    
                    if frame_id in frame_annotations:
                        # Add sample to our dataset
                        samples.append((img_path, frame_annotations[frame_id]))
            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, annotations = self.samples[idx]
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Convert annotations to the format expected by the model
        boxes = []
        labels = []
        
        for ann in annotations:
            x, y, w, h, category = ann
            
            # Convert from [x, y, w, h] to [x1, y1, x2, y2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
                
            boxes.append([x1, y1, x2, y2])
            labels.append(category - 1)  # Adjust category to be 0-indexed
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transformations if specified
        if self.transform:
            # Albumentations expects boxes in normalized format
            if len(boxes) > 0:
                normalized_boxes = boxes.copy()
                normalized_boxes[:, [0, 2]] /= width
                normalized_boxes[:, [1, 3]] /= height
                
                transformed = self.transform(
                    image=image,
                    bboxes=normalized_boxes,
                    labels=labels
                )
                
                image = transformed['image']
                
                # Convert back to pixel coordinates
                transformed_boxes = np.array(transformed['bboxes'], dtype=np.float32)
                if len(transformed_boxes) > 0:
                    transformed_boxes[:, [0, 2]] *= width
                    transformed_boxes[:, [1, 3]] *= height
                    boxes = transformed_boxes
                    labels = np.array(transformed['labels'], dtype=np.int64)
                else:
                    # In case all boxes were removed by transformations
                    boxes = np.zeros((0, 4), dtype=np.float32)
                    labels = np.zeros((0), dtype=np.int64)
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
        else:
            # Convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
        # If no boxes, use empty arrays
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0), dtype=np.int64)
            
        # Create target dictionary
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }
        
        return image, target


def get_transform(train):
    """
    Get transformations for training and validation
    """
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(height=640, width=640, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(height=640, width=640, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def create_data_loaders(batch_size=8):
    """
    Create data loaders for training and validation
    """
    # Create train dataset
    train_dataset = VisDroneDataset(
        sequences_dir=config.SEQUENCES_PATH,
        annotations_dir=config.ANNOTATIONS_PATH,
        transform=get_transform(train=True),
        train=True,
        validation_split=config.VALIDATION_SPLIT,
        seed=config.SEED
    )
    
    # Create validation dataset
    val_dataset = VisDroneDataset(
        sequences_dir=config.SEQUENCES_PATH,
        annotations_dir=config.ANNOTATIONS_PATH,
        transform=get_transform(train=False),
        train=False,
        validation_split=config.VALIDATION_SPLIT,
        seed=config.SEED
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def collate_fn(batch):
    """
    Custom collate function to handle variable size images and targets
    """
    images, targets = zip(*batch)
    return list(images), list(targets) 