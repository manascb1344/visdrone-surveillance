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
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from PIL import Image
import shutil

class YOLOFormatDataset(Dataset):
    """Dataset class to load YOLO format data for Faster R-CNN training"""
    
    def __init__(self, images_dir, labels_dir, img_size=640, class_names=None, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.transform = transform
        self.class_names = class_names or [
            'person', 'bicycle', 'car', 'van', 
            'truck', 'tricycle', 'bus', 'motorcycle'
        ]
        
        # Get all image files
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        # Load image
        img_name = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        height, width = image.shape[:2]
        
        # Resize image if needed
        if max(height, width) != self.img_size:
            # Maintain aspect ratio
            if height > width:
                new_height = self.img_size
                new_width = int(width * (self.img_size / height))
            else:
                new_width = self.img_size
                new_height = int(height * (self.img_size / width))
            
            image = cv2.resize(image, (new_width, new_height))
        
        # Convert to PIL Image for torchvision transforms
        image = Image.fromarray(image)
        
        # Get label file path
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + '.txt')
        
        boxes = []
        labels = []
        
        # Read label file if it exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) < 5:
                        continue
                    
                    # YOLO format: class x_center y_center width height (normalized)
                    class_id = int(data[0])
                    x_center = float(data[1]) * width
                    y_center = float(data[2]) * height
                    w = float(data[3]) * width
                    h = float(data[4]) * height
                    
                    # Convert to (x1, y1, x2, y2) format for Faster R-CNN
                    x1 = max(0, x_center - w / 2)
                    y1 = max(0, y_center - h / 2)
                    x2 = min(width, x_center + w / 2)
                    y2 = min(height, y_center + h / 2)
                    
                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id + 1)  # Add 1 because 0 is background in Faster RCNN
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Basic transform: convert to tensor and normalize
            image = torchvision.transforms.ToTensor()(image)
        
        return image, target

class FasterRCNNTrainer:
    """Class to train and manage Faster R-CNN models using YOLO formatted datasets"""
    
    def __init__(self, project_dir="./visdrone_faster_rcnn_training"):
        self.project_dir = project_dir
        os.makedirs(project_dir, exist_ok=True)
        
        # Default class names for VisDrone dataset
        self.class_names = [
            'person', 'bicycle', 'car', 'van', 
            'truck', 'tricycle', 'bus', 'motorcycle'
        ]
    
    def build_model(self, num_classes):
        # Load a pre-trained Faster R-CNN model
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        
        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained head with a new one for our number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)  # +1 for background
        
        return model
    
    def train_faster_rcnn(self, 
                         dataset_yaml, 
                         epochs=50, 
                         batch_size=16, 
                         img_size=640, 
                         num_gpus=1,
                         learning_rate=0.001):
        
        # Load dataset configuration
        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        dataset_path = dataset_config['path']
        self.class_names = dataset_config['names']
        
        # Set device
        if torch.cuda.is_available() and num_gpus > 0:
            device = torch.device('cuda')
            print(f"Using {num_gpus} GPU(s) for training")
        else:
            device = torch.device('cpu')
            print("No GPU available, using CPU for training (this will be slow)")
        
        # Dataset paths
        train_images = os.path.join(dataset_path, dataset_config['train'])
        train_labels = os.path.join(dataset_path, 'labels', 'train')
        val_images = os.path.join(dataset_path, dataset_config['val'])
        val_labels = os.path.join(dataset_path, 'labels', 'val')
        
        # Create datasets
        train_dataset = YOLOFormatDataset(
            images_dir=train_images,
            labels_dir=train_labels,
            img_size=img_size,
            class_names=self.class_names
        )
        
        val_dataset = YOLOFormatDataset(
            images_dir=val_images,
            labels_dir=val_labels,
            img_size=img_size,
            class_names=self.class_names
        )
        
        print(f"Training dataset: {len(train_dataset)} images")
        print(f"Validation dataset: {len(val_dataset)} images")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x)),  # Necessary for variable sized boxes
            num_workers=min(8, os.cpu_count() or 4)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: tuple(zip(*x)),
            num_workers=min(8, os.cpu_count() or 4)
        )
        
        # Initialize model
        num_classes = len(self.class_names)
        model = self.build_model(num_classes)
        model.to(device)
        
        # Multi-GPU support
        if torch.cuda.is_available() and num_gpus > 1:
            model = nn.DataParallel(model)
        
        # Set up optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
        
        # Learning rate scheduler
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Run name for storing results
        run_name = f"faster_rcnn_visdrone_{num_gpus}GPU" if num_gpus > 0 else "faster_rcnn_visdrone_CPU"
        run_dir = os.path.join(self.project_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # Create subdirectories
        weights_dir = os.path.join(run_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        # Training loop
        best_map = 0.0
        training_results = {"epoch": [], "loss": [], "val_map": []}
        
        print(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Set model to training mode
            model.train()
            epoch_loss = 0.0
            
            # Progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for images, targets in pbar:
                # Move to device
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = model(images, targets)
                
                # Calculate total loss
                losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                # Update progress bar
                loss_value = losses.item()
                epoch_loss += loss_value
                pbar.set_postfix({"loss": loss_value})
            
            # Update learning rate scheduler
            lr_scheduler.step()
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Evaluate on validation set
            model.eval()
            val_map = self._evaluate(model, val_loader, device)
            print(f"Validation mAP@0.5: {val_map:.4f}")
            
            # Record results
            training_results["epoch"].append(epoch + 1)
            training_results["loss"].append(avg_loss)
            training_results["val_map"].append(val_map)
            
            # Save results to CSV
            with open(os.path.join(run_dir, "results.csv"), "w") as f:
                f.write("epoch,loss,val_map\n")
                for e, l, m in zip(training_results["epoch"], training_results["loss"], training_results["val_map"]):
                    f.write(f"{e},{l},{m}\n")
            
            # Save checkpoint
            checkpoint_path = os.path.join(weights_dir, f"epoch_{epoch+1}.pt")
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), checkpoint_path)
            
            # Save best model
            if val_map > best_map:
                best_map = val_map
                best_model_path = os.path.join(weights_dir, "best.pt")
                torch.save(model_to_save.state_dict(), best_model_path)
                print(f"New best model saved with mAP: {best_map:.4f}")
        
        # Training completed
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time/60:.2f} minutes")
        
        # Plot training results
        self._plot_training_results(training_results, run_dir)
        
        # Return results
        return {
            "model": model,
            "best_map": best_map,
            "training_time": elapsed_time,
            "best_model_path": os.path.join(weights_dir, "best.pt"),
            "run_dir": run_dir
        }
    
    def _evaluate(self, model, data_loader, device):
        # This is a simplified mAP calculation - in a real implementation, 
        # you would use a proper detection evaluation toolkit like pycocotools
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                
                outputs = model(images)
                
                for i, output in enumerate(outputs):
                    # Get target boxes and labels
                    target_boxes = targets[i]['boxes'].cpu().numpy()
                    target_labels = targets[i]['labels'].cpu().numpy()
                    
                    # Get predicted boxes, scores, and labels
                    pred_boxes = output['boxes'].cpu().numpy()
                    pred_scores = output['scores'].cpu().numpy()
                    pred_labels = output['labels'].cpu().numpy()
                    
                    all_predictions.append({
                        'boxes': pred_boxes,
                        'scores': pred_scores,
                        'labels': pred_labels
                    })
                    
                    all_targets.append({
                        'boxes': target_boxes,
                        'labels': target_labels
                    })
        
        # Calculate mAP@0.5
        map_50 = self._calculate_map(all_predictions, all_targets, iou_threshold=0.5)
        return map_50
    
    def _calculate_map(self, all_predictions, all_targets, iou_threshold=0.5):
        """
        Calculate mean Average Precision (mAP) at specified IoU threshold
        This is a simplified version - a real implementation would use COCO evaluation
        """
        if not all_predictions:
            return 0.0
        
        def calculate_iou(box1, box2):
            # Calculate IoU between two boxes in format [x1, y1, x2, y2]
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            # Calculate overlap area
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            intersection = w * h
            
            # Calculate union area
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = box1_area + box2_area - intersection
            
            return intersection / union if union > 0 else 0
        
        # Calculate AP for each class
        unique_classes = set()
        for target in all_targets:
            unique_classes.update(target['labels'])
        
        average_precisions = []
        
        for class_id in unique_classes:
            true_positives = []
            false_positives = []
            scores = []
            
            for i, (preds, targets) in enumerate(zip(all_predictions, all_targets)):
                pred_boxes = preds['boxes']
                pred_scores = preds['scores']
                pred_labels = preds['labels']
                
                target_boxes = targets['boxes']
                target_labels = targets['labels']
                
                # Get predictions for this class
                class_pred_indices = [j for j, label in enumerate(pred_labels) if label == class_id]
                class_pred_boxes = pred_boxes[class_pred_indices] if len(class_pred_indices) > 0 else []
                class_pred_scores = pred_scores[class_pred_indices] if len(class_pred_indices) > 0 else []
                
                # Get targets for this class
                class_target_indices = [j for j, label in enumerate(target_labels) if label == class_id]
                class_target_boxes = target_boxes[class_target_indices] if len(class_target_indices) > 0 else []
                
                # Mark targets as "not detected yet"
                detected_targets = [False] * len(class_target_boxes)
                
                # Sort predictions by score (descending)
                if len(class_pred_scores) > 0:
                    sorted_indices = np.argsort(-class_pred_scores)
                    class_pred_boxes = class_pred_boxes[sorted_indices]
                    class_pred_scores = class_pred_scores[sorted_indices]
                
                # For each prediction, check if it matches a target
                for pred_idx, pred_box in enumerate(class_pred_boxes):
                    if len(class_target_boxes) == 0:
                        false_positives.append(1)
                        true_positives.append(0)
                        scores.append(class_pred_scores[pred_idx])
                        continue
                    
                    # Find best matching target
                    best_iou = 0
                    best_target_idx = -1
                    
                    for target_idx, target_box in enumerate(class_target_boxes):
                        if detected_targets[target_idx]:
                            continue
                        
                        iou = calculate_iou(pred_box, target_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_target_idx = target_idx
                    
                    # If IoU > threshold, it's a true positive
                    if best_iou >= iou_threshold and best_target_idx != -1:
                        true_positives.append(1)
                        false_positives.append(0)
                        detected_targets[best_target_idx] = True
                    else:
                        true_positives.append(0)
                        false_positives.append(1)
                    
                    scores.append(class_pred_scores[pred_idx])
            
            # Calculate precision and recall
            if not scores:
                continue
                
            # Convert to numpy arrays
            scores = np.array(scores)
            true_positives = np.array(true_positives)
            false_positives = np.array(false_positives)
            
            # Sort by score
            sorted_indices = np.argsort(-scores)
            true_positives = true_positives[sorted_indices]
            false_positives = false_positives[sorted_indices]
            
            # Compute cumulative true positives and false positives
            cumulative_tp = np.cumsum(true_positives)
            cumulative_fp = np.cumsum(false_positives)
            
            # Calculate precision and recall
            precision = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-10)
            
            # Calculate total number of ground truth boxes for this class
            total_gt = sum(len([box for box, label in zip(target['boxes'], target['labels']) if label == class_id]) 
                          for target in all_targets)
            
            recall = cumulative_tp / (total_gt + 1e-10)
            
            # Compute AP by computing the area under the precision-recall curve
            # 11-point interpolation method
            ap = 0
            for t in np.linspace(0, 1, 11):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                ap += p / 11
            
            average_precisions.append(ap)
        
        return np.mean(average_precisions) if average_precisions else 0.0
    
    def _plot_training_results(self, results, run_dir):
        # Plot training loss and validation mAP
        epochs = results["epoch"]
        
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, results["loss"])
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        
        # mAP plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, results["val_map"])
        plt.title("Validation mAP@0.5")
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "training_results.png"))
        plt.close()
    
    def save_model(self, model_path, save_dir, new_name=None):
        os.makedirs(save_dir, exist_ok=True)
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
            
        if new_name:
            target_path = os.path.join(save_dir, new_name)
        else:
            target_path = os.path.join(save_dir, os.path.basename(model_path))
            
        try:
            shutil.copy(model_path, target_path)
            print(f"Model saved to: {target_path}")
            return target_path
        except Exception as e:
            print(f"Error saving model: {e}")
            return None

class FasterRCNNWrapper:
    """
    Wrapper class to adapt Faster R-CNN model to YOLO-like inference
    This allows Faster R-CNN to be used in ensemble with YOLO models
    """
    
    def __init__(self, model_path, class_names=None):
        self.model_path = model_path
        self.class_names = class_names or [
            'person', 'bicycle', 'car', 'van', 
            'truck', 'tricycle', 'bus', 'motorcycle'
        ]
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path, len(self.class_names))
        
        print(f"Faster R-CNN model loaded from: {model_path}")
    
    def _load_model(self, model_path, num_classes):
        model = fasterrcnn_resnet50_fpn(weights=None)
        
        # Get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the head
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)  # +1 for background
        
        # Load saved weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, img, conf=0.25, verbose=False):
        """
        YOLO-like predict method to maintain compatibility with the ensemble system
        """
        # Handle different input types
        if isinstance(img, str):  # Path to image
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
            if img.dtype == np.uint8 and img[0,0,0] > 1:  # BGR to RGB if needed
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        original_height, original_width = img.shape[:2]
        
        # Prepare image for model
        img_tensor = torchvision.transforms.ToTensor()(img).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            prediction = self.model([img_tensor])[0]
        
        # Filter predictions by confidence
        keep = prediction['scores'] >= conf
        boxes = prediction['boxes'][keep]
        scores = prediction['scores'][keep]
        labels = prediction['labels'][keep] - 1  # Subtract 1 to convert from RCNN classes (0=background) to YOLO format
        
        # Create result objects similar to YOLO's format for compatibility
        class Results:
            def __init__(self, boxes, scores, labels, orig_shape):
                self.orig_shape = orig_shape
                
                class Boxes:
                    def __init__(self, boxes, scores, labels):
                        self.xyxy = boxes
                        self.conf = scores
                        self.cls = labels
                        self.data = torch.cat((boxes, 
                                              torch.zeros(len(boxes), 1),  # Add dummy track_id column
                                              scores.unsqueeze(1),
                                              labels.unsqueeze(1)), dim=1)
                        self.id = None
                
                self.boxes = Boxes(boxes, scores, labels) if len(boxes) > 0 else None
        
        result = Results(boxes, scores, labels, (original_height, original_width))
        
        return [result]
    
    # For compatibility with YOLO interface when used in ensemble
    def track(self, img, persist=True, conf=0.25, verbose=False):
        """Return same format as predict since tracking is not supported"""
        return self.predict(img, conf, verbose) 