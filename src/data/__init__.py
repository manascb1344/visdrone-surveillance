"""
Data handling utilities for VisDrone object detection.
"""

from pathlib import Path
import os

# Add path helpers
def get_dataset_path(dataset_name="VisDrone-YOLO"):
    """Return the path to the dataset directory."""
    project_root = Path(__file__).resolve().parents[2]  # Go up two directories from this file
    return project_root / "datasets" / dataset_name

def get_config_path(config_name="visdrone.yaml"):
    """Return the path to the configuration file."""
    project_root = Path(__file__).resolve().parents[2]  # Go up two directories from this file
    return project_root / "configs" / config_name
