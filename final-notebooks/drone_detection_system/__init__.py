"""
Drone Detection System Package

A system for drone-based object detection using YOLOv8 and the VisDrone dataset.
"""

__version__ = '1.0.0'

from .data_manager import DataManager
from .model_trainer import ModelTrainer
from .visualization import Visualizer
from .rtmp_processor import RTMPProcessor 