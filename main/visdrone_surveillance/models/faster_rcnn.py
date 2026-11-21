import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys
import os

from .. import config

def get_model(num_classes):
    """
    Load a pre-trained Faster R-CNN model and modify it for our dataset
    
    Args:
        num_classes (int): Number of object classes to detect (including background)
    
    Returns:
        model: The modified Faster R-CNN model
    """
    # Load pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def save_model(model, path):
    """
    Save model to disk
    
    Args:
        model: PyTorch model to save
        path (str): Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path, num_classes):
    """
    Load model from disk
    
    Args:
        path (str): Path to the model
        num_classes (int): Number of object classes
    
    Returns:
        model: The loaded model
    """
    # Get a new model
    model = get_model(num_classes)
    
    # Load the saved state dict
    model.load_state_dict(torch.load(path))
    
    return model 