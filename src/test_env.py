#!/usr/bin/env python
"""
Environment Test Script
This script verifies that all required libraries for the drone project
can be imported correctly.
"""

# Import standard libraries
import os
import sys
import time
from pathlib import Path

# Test PyTorch imports
print("Testing PyTorch...")
import torch
import torchvision
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Test OpenCV import
print("\nTesting OpenCV...")
import cv2
print(f"OpenCV version: {cv2.__version__}")

# Test YOLO import
print("\nTesting Ultralytics YOLO...")
from ultralytics import YOLO
print("Ultralytics YOLO imported successfully")

# Test data processing libraries
print("\nTesting data processing libraries...")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
print("All data processing libraries imported successfully")

# Test API libraries
print("\nTesting API libraries...")
import fastapi
print("FastAPI imported successfully")

print("\nAll libraries imported successfully. Environment is correctly set up.") 