"""Configuration for the VisDrone surveillance package.

This module is the single source of truth for dataset paths, class labels,
training defaults, and RTMP settings. Values can be overridden through
environment variables (see below) without changing source code.

Environment variables:
    VISDRONE_DATASET_ROOT   VisDrone VID root directory (Faster R-CNN training)
    VISDRONE_YOLO_ROOT      VisDrone YOLO root directory (YOLO training)
    RTMP_SERVER             RTMP server URL
    RTMP_STREAM_KEY         RTMP stream key
    CONFIDENCE_THRESHOLD    Default detection confidence threshold
    NMS_THRESHOLD           Default NMS IoU threshold
"""

import os
from pathlib import Path

# main/visdrone_surveillance/config.py -> main/visdrone_surveillance -> main -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset paths (VisDrone VID format, used by Faster R-CNN training)
DATASET_ROOT = os.getenv(
    "VISDRONE_DATASET_ROOT",
    str(PROJECT_ROOT / "datasets" / "VisDrone2019-VID-train"),
)
SEQUENCES_PATH = os.path.join(DATASET_ROOT, "sequences")
ANNOTATIONS_PATH = os.path.join(DATASET_ROOT, "annotations")

# Dataset paths (VisDrone YOLO format, used by YOLO training)
YOLO_DATASET_ROOT = os.getenv(
    "VISDRONE_YOLO_ROOT",
    str(PROJECT_ROOT / "datasets" / "VisDrone-YOLO"),
)

# Training parameters
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
VALIDATION_SPLIT = 0.2
SEED = 42

# Model parameters
NUM_CLASSES = 10  # Number of object categories in VisDrone (stale; len(CLASSES) is 12)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
NMS_THRESHOLD = float(os.getenv("NMS_THRESHOLD", "0.4"))

# RTMP settings
RTMP_SERVER = os.getenv("RTMP_SERVER", "rtmp://localhost/live/stream")
RTMP_STREAM_KEY = os.getenv("RTMP_STREAM_KEY", "")
RTMP_OUTPUT_URL = f"{RTMP_SERVER}/{RTMP_STREAM_KEY}"

# Object classes in VisDrone (category index -> class name, category 0 is ignored)
CLASSES = [
    "ignored regions",
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    "others",
]
