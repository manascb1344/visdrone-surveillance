import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import yaml
import cv2
import numpy as np
from PIL import Image

class YOLOFormatDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640, class_names=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.class_names = class_names or [
            'person', 'bicycle', 'car', 'van', 
            'truck', 'tricycle', 'bus', 'motorcycle'
        ]
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_name = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        if max(height, width) != self.img_size:
            if height > width:
                new_height = self.img_size
                new_width = int(width * (self.img_size / height))
            else:
                new_width = self.img_size
                new_height = int(height * (self.img_size / width))
            image = cv2.resize(image, (new_width, new_height))
        
        image = Image.fromarray(image)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + '.txt')
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) < 5: continue
                    
                    class_id = int(data[0])
                    x_center = float(data[1]) * width
                    y_center = float(data[2]) * height
                    w = float(data[3]) * width
                    h = float(data[4]) * height
                    
                    x1 = max(0, x_center - w / 2)
                    y1 = max(0, y_center - h / 2)
                    x2 = min(width, x_center + w / 2)
                    y2 = min(height, y_center + h / 2)
                    
                    if x2 <= x1 or y2 <= y1: continue
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id + 1)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        image = torchvision.transforms.ToTensor()(image)
        return image, target

class FasterRCNNTrainer:
    def __init__(self, project_dir="./visdrone_faster_rcnn_training"):
        self.project_dir = project_dir
        os.makedirs(project_dir, exist_ok=True)
        self.class_names = ['person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motorcycle']
    
    def build_model(self, num_classes):
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
        return model
    
    def train_faster_rcnn(self, dataset_yaml, epochs=50, batch_size=16, img_size=640, num_gpus=1, learning_rate=0.001):
        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        dataset_path = dataset_config['path']
        self.class_names = dataset_config['names']
        device = torch.device('cuda' if torch.cuda.is_available() and num_gpus > 0 else 'cpu')
        
        print(f"Loading dataset from {dataset_path}")
        train_dataset = YOLOFormatDataset(
            images_dir=os.path.join(dataset_path, dataset_config['train']),
            labels_dir=os.path.join(dataset_path, 'labels', 'train'),
            img_size=img_size,
            class_names=self.class_names
        )
        
        print(f"Creating data loader with batch size {batch_size}")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x)),
            num_workers=min(8, os.cpu_count() or 4)
        )
        
        print("Building model...")
        model = self.build_model(len(self.class_names))
        model.to(device)
        print(f"Model moved to device: {device}")
        
        if torch.cuda.is_available() and num_gpus > 1:
            model = nn.DataParallel(model)
            print(f"Using {num_gpus} GPUs for training")
        
        optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], 
                            lr=learning_rate, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            batch_count = 0
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            for i, (images, targets) in enumerate(train_loader):
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                epoch_loss += losses.item()
                batch_count += 1
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                if (i + 1) % 10 == 0:  # Print every 10 batches
                    print(f"  Batch {i+1}/{len(train_loader)}, Loss: {losses.item():.4f}")
            
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            lr_scheduler.step()
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(self.project_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
        
        print("Training completed!")
        return model

class FasterRCNNWrapper:
    def __init__(self, model_path, class_names=None):
        self.class_names = class_names or ['person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motorcycle']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path, len(self.class_names))
    
    def _load_model(self, model_path, num_classes):
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, img, conf=0.25):
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
            if img.dtype == np.uint8 and img[0,0,0] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        original_height, original_width = img.shape[:2]
        img_tensor = torchvision.transforms.ToTensor()(img).to(self.device)
        
        with torch.no_grad():
            prediction = self.model([img_tensor])[0]
            
            # --- Debugging Start ---
            print(f"[FasterRCNNWrapper Debug] Raw Prediction Keys: {prediction.keys()}")
            if 'scores' in prediction:
                print(f"[FasterRCNNWrapper Debug] Raw Scores Count: {len(prediction['scores'])}")
                if len(prediction['scores']) > 0:
                    print(f"[FasterRCNNWrapper Debug] Max Raw Score: {torch.max(prediction['scores'])}")
                    print(f"[FasterRCNNWrapper Debug] Raw Scores Sample: {prediction['scores'][:5]}") # Print first 5 scores
            if 'boxes' in prediction:
                print(f"[FasterRCNNWrapper Debug] Raw Boxes Count: {len(prediction['boxes'])}")
            if 'labels' in prediction:
                print(f"[FasterRCNNWrapper Debug] Raw Labels Count: {len(prediction['labels'])}")
            # --- Debugging End ---

        keep = prediction['scores'] >= conf
        boxes = prediction['boxes'][keep]
        scores = prediction['scores'][keep]
        labels = prediction['labels'][keep] - 1
        
        class Results:
            def __init__(self, boxes, scores, labels, orig_shape, device):
                self.orig_shape = orig_shape
                self.device = device  # Store device as an instance attribute
                
                class Boxes:
                    def __init__(self, boxes, scores, labels, device):
                        self.xyxy = boxes
                        self.conf = scores
                        self.cls = labels
                        if len(boxes) > 0:
                            self.data = torch.cat((boxes, torch.zeros(len(boxes), 1, device=device), 
                                                 scores.unsqueeze(1), labels.unsqueeze(1)), dim=1)
                        else:
                            # Handle empty detection case
                            self.data = torch.zeros((0, 7), device=device)
                        self.id = None
                
                # Always create a Boxes object, even with empty detections
                self.boxes = Boxes(boxes, scores, labels, device)
        
        # Pass the device to the Results constructor
        return [Results(boxes, scores, labels, (original_height, original_width), self.device)]
    
    def track(self, img, persist=True, conf=0.25):
        return self.predict(img, conf)