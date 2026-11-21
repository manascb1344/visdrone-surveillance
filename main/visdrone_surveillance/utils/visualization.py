import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import sys
import os

from .. import config

def draw_boxes(image, boxes, labels, scores=None, class_names=None, threshold=0.5):
    """
    Draw bounding boxes on an image
    
    Args:
        image (numpy.ndarray): Image to draw on
        boxes (torch.Tensor or numpy.ndarray): Bounding boxes with shape [N, 4]
        labels (torch.Tensor or numpy.ndarray): Labels with shape [N]
        scores (torch.Tensor or numpy.ndarray, optional): Confidence scores with shape [N]
        class_names (list, optional): List of class names
        threshold (float, optional): Confidence threshold
    
    Returns:
        numpy.ndarray: Image with bounding boxes
    """
    # Convert tensors to numpy if needed
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    # Convert image to RGB if in BGR format (common with OpenCV)
    if image.shape[2] == 3 and np.max(image) <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Create a copy to avoid modifying the original
    image_copy = image.copy()
    
    # Get image dimensions
    height, width = image_copy.shape[:2]
    
    # Define color map for different classes
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Green (dark)
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
    ]
    
    # Draw each bounding box
    for i, (box, label) in enumerate(zip(boxes, labels)):
        # Skip if confidence is below threshold
        if scores is not None and scores[i] < threshold:
            continue
        
        # Get coordinates
        x1, y1, x2, y2 = box.astype(int)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # Get color for this class
        color = colors[label % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        if class_names is not None:
            label_text = class_names[label]
        else:
            label_text = f"Class {label}"
            
        if scores is not None:
            label_text += f": {scores[i]:.2f}"
            
        # Draw label background
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            image_copy, 
            (x1, y1 - text_size[1] - 5), 
            (x1 + text_size[0], y1), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            image_copy,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    return image_copy


def visualize_predictions(model, images, targets, device, class_names=None, threshold=0.5, figsize=(15, 10)):
    """
    Visualize model predictions alongside ground truth
    
    Args:
        model: PyTorch detection model
        images (list): List of images
        targets (list): List of targets
        device: Device to run the model on
        class_names (list, optional): List of class names
        threshold (float, optional): Confidence threshold
        figsize (tuple, optional): Figure size
    
    Returns:
        matplotlib.figure.Figure: Figure with visualizations
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Process each image
    num_images = min(len(images), 4)  # Show at most 4 images
    
    with torch.no_grad():
        for i in range(num_images):
            # Clone the image and move to device
            image = images[i].to(device)
            
            # Run forward pass
            predictions = model([image])
            
            # Get predictions
            pred_boxes = predictions[0]['boxes'].cpu()
            pred_labels = predictions[0]['labels'].cpu()
            pred_scores = predictions[0]['scores'].cpu()
            
            # Convert image for visualization
            image = image.cpu().permute(1, 2, 0).numpy()
            
            # Get ground truth
            gt_boxes = targets[i]['boxes'].cpu()
            gt_labels = targets[i]['labels'].cpu()
            
            # Create subplots
            ax1 = fig.add_subplot(num_images, 2, i*2 + 1)
            ax2 = fig.add_subplot(num_images, 2, i*2 + 2)
            
            # Draw ground truth boxes
            gt_image = draw_boxes(
                image.copy(), 
                gt_boxes, 
                gt_labels, 
                class_names=class_names
            )
            
            # Draw predicted boxes
            pred_image = draw_boxes(
                image.copy(), 
                pred_boxes, 
                pred_labels, 
                scores=pred_scores, 
                class_names=class_names, 
                threshold=threshold
            )
            
            # Display images
            ax1.imshow(gt_image)
            ax1.set_title(f"Ground Truth")
            ax1.axis('off')
            
            ax2.imshow(pred_image)
            ax2.set_title(f"Predictions (conf >= {threshold})")
            ax2.axis('off')
    
    plt.tight_layout()
    return fig


def plot_metrics(metrics):
    """
    Plot evaluation metrics
    
    Args:
        metrics (dict): Dictionary of metrics from evaluation
    
    Returns:
        matplotlib.figure.Figure: Figure with plots
    """
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot mAP per class
    ax1 = fig.add_subplot(2, 2, 1)
    class_ids = sorted(metrics['ap_per_class'].keys())
    ap_values = [metrics['ap_per_class'][class_id] for class_id in class_ids]
    
    ax1.bar(range(len(class_ids)), ap_values)
    ax1.set_xticks(range(len(class_ids)))
    ax1.set_xticklabels([config.CLASSES[class_id + 1] for class_id in class_ids], rotation=45, ha='right')
    ax1.set_ylabel('Average Precision')
    ax1.set_title(f'AP per class (mAP: {metrics["mAP"]:.4f})')
    
    # Plot precision-recall curves for top classes
    ax2 = fig.add_subplot(2, 2, 2)
    for class_id in class_ids[:5]:  # Show top 5 classes
        if class_id in metrics['precision'] and class_id in metrics['recall']:
            ax2.plot(
                metrics['recall'][class_id], 
                metrics['precision'][class_id], 
                label=f"{config.CLASSES[class_id + 1]} (AP: {metrics['ap_per_class'][class_id]:.4f})"
            )
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend()
    
    # Plot ground truth counts per class
    ax3 = fig.add_subplot(2, 2, 3)
    gt_counts = [metrics['total_gt'].get(class_id, 0) for class_id in class_ids]
    
    ax3.bar(range(len(class_ids)), gt_counts)
    ax3.set_xticks(range(len(class_ids)))
    ax3.set_xticklabels([config.CLASSES[class_id + 1] for class_id in class_ids], rotation=45, ha='right')
    ax3.set_ylabel('Count')
    ax3.set_title('Ground Truth Counts per Class')
    
    # Plot true positive rate per class
    ax4 = fig.add_subplot(2, 2, 4)
    tp_rates = []
    
    for class_id in class_ids:
        tp = metrics['true_positives'].get(class_id, 0)
        total_pred = metrics['total_pred'].get(class_id, 0)
        tp_rate = tp / total_pred if total_pred > 0 else 0
        tp_rates.append(tp_rate)
    
    ax4.bar(range(len(class_ids)), tp_rates)
    ax4.set_xticks(range(len(class_ids)))
    ax4.set_xticklabels([config.CLASSES[class_id + 1] for class_id in class_ids], rotation=45, ha='right')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('True Positive Rate per Class')
    
    plt.tight_layout()
    return fig 