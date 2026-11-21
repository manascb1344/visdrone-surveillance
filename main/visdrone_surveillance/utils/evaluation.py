import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import sys
import os

from .. import config

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1 (torch.Tensor): First bounding box [x1, y1, x2, y2]
        box2 (torch.Tensor): Second bounding box [x1, y1, x2, y2]
    
    Returns:
        float: IoU value
    """
    # Get coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union
    union = box1_area + box2_area - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0
    
    return intersection / union


def calculate_ap(recall, precision):
    """
    Calculate Average Precision (AP) given precision and recall values
    
    Args:
        recall (list): Recall values
        precision (list): Precision values
    
    Returns:
        float: Average Precision value
    """
    # Convert to numpy arrays and ensure they start and end properly
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))
    
    # Ensure precision is non-increasing from right to left
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])
    
    # Calculate areas under the precision curve at each recall point
    ap = 0
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    for i in indices:
        ap += (recall[i] - recall[i-1]) * precision[i]
    
    return ap


def evaluate_model(model, data_loader, device, iou_threshold=0.5, confidence_threshold=0.5):
    """
    Evaluate model performance on a dataset
    
    Args:
        model: PyTorch detection model
        data_loader: DataLoader for the validation dataset
        device: Device to run the model on (CPU or GPU)
        iou_threshold (float): IoU threshold for considering a detection as correct
        confidence_threshold (float): Confidence threshold for considering a detection
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    metrics = {
        'ap_per_class': defaultdict(float),
        'precision': defaultdict(list),
        'recall': defaultdict(list),
        'total_gt': defaultdict(int),
        'total_pred': defaultdict(int),
        'true_positives': defaultdict(int),
    }
    
    # Store detections for AP calculation
    all_detections = defaultdict(list)  # {class_id: [[image_idx, confidence, iou_with_gt]]}
    num_gts = defaultdict(int)  # {class_id: count}
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
            # Move images to device
            images = [img.to(device) for img in images]
            
            # Run forward pass
            predictions = model(images)
            
            # Process each image in the batch
            for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
                # Get ground truth boxes and labels
                gt_boxes = target['boxes'].to(device)
                gt_labels = target['labels'].to(device)
                
                # Get predicted boxes, labels, and scores
                pred_boxes = pred['boxes']
                pred_labels = pred['labels']
                pred_scores = pred['scores']
                
                # Filter predictions by confidence threshold
                keep = pred_scores > confidence_threshold
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]
                
                # Count ground truths per class
                for gt_label in gt_labels:
                    num_gts[gt_label.item()] += 1
                
                # Mark each prediction as true positive or false positive
                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    label = label.item()
                    metrics['total_pred'][label] += 1
                    
                    # Find ground truth boxes of this class
                    gt_class_boxes = gt_boxes[gt_labels == label]
                    
                    if len(gt_class_boxes) == 0:
                        # No ground truth of this class in the image
                        all_detections[label].append([i * len(images) + img_idx, score.item(), 0])
                        continue
                    
                    # Calculate IoU with all ground truth boxes of this class
                    ious = torch.stack([calculate_iou(box, gt_box) for gt_box in gt_class_boxes])
                    
                    # Check if any IoU exceeds the threshold
                    if torch.max(ious) >= iou_threshold:
                        # This is a true positive
                        metrics['true_positives'][label] += 1
                        # Store detection with highest IoU
                        best_iou = torch.max(ious).item()
                        all_detections[label].append([i * len(images) + img_idx, score.item(), best_iou])
                    else:
                        # This is a false positive
                        all_detections[label].append([i * len(images) + img_idx, score.item(), 0])
    
    # Calculate AP for each class
    for class_id in all_detections:
        # Skip if no ground truth for this class
        if num_gts[class_id] == 0:
            metrics['ap_per_class'][class_id] = 0
            continue
        
        # Sort detections by confidence
        detections = sorted(all_detections[class_id], key=lambda x: x[1], reverse=True)
        
        # Calculate precision and recall values
        tp = 0
        fp = 0
        precision = []
        recall = []
        
        for _, _, iou in detections:
            if iou >= iou_threshold:
                tp += 1
            else:
                fp += 1
            
            precision.append(tp / (tp + fp))
            recall.append(tp / num_gts[class_id])
        
        # Calculate AP
        metrics['ap_per_class'][class_id] = calculate_ap(recall, precision)
        metrics['precision'][class_id] = precision
        metrics['recall'][class_id] = recall
        metrics['total_gt'][class_id] = num_gts[class_id]
    
    # Calculate mAP (mean Average Precision)
    metrics['mAP'] = sum(metrics['ap_per_class'].values()) / len(metrics['ap_per_class']) if metrics['ap_per_class'] else 0
    
    return metrics 