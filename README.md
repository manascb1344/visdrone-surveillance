# VisDrone-RTMP: Real-Time Drone Object Detection System

A real-time object detection system for drone video feeds using the VisDrone dataset, optimized for aerial imagery.

## Overview

This project implements a system for performing real-time object detection on live video feeds from drones via RTMP streaming. The system is designed to identify and track objects of interest (primarily vehicles and people) from aerial perspectives.

The project utilizes the VisDrone dataset for training and implements two different object detection models:
- A single-stage detector (YOLO-based)
- A two-stage detector (Faster R-CNN-based)

## Key Features

- Download and preprocess the VisDrone dataset specifically for aerial object detection
- Implement data augmentation techniques specialized for aerial imagery
- Train two different object detection models (YOLO and Faster R-CNN) on the VisDrone dataset
- Ingest and decode RTMP video streams from drones in real-time
- Perform real-time object detection on incoming video frames
- Visualize detection results by drawing bounding boxes and labels
- Compare performance metrics between different model architectures

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- FFmpeg installed

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/visdrone-rtmp.git
   cd visdrone-rtmp
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the VisDrone dataset (instructions in the documentation)

## Project Structure

```
visdrone-rtmp/
├── data/                  # Data directory (created on first run)
│   ├── raw/               # Raw VisDrone dataset
│   └── processed/         # Processed dataset
├── models/                # Model implementations
│   ├── yolo/              # YOLO-based model
│   └── faster_rcnn/       # Faster R-CNN model
├── src/
│   ├── data/              # Data processing utilities
│   ├── stream/            # RTMP stream handling
│   ├── detection/         # Object detection modules
│   └── visualization/     # Visualization utilities
├── notebooks/             # Jupyter notebooks for experimentation
├── tests/                 # Unit and integration tests
├── scripts/               # Utility scripts
└── output/                # For saving model outputs
```

## Usage

1. Process the VisDrone dataset:
   ```
   python scripts/process_dataset.py
   ```

2. Train models:
   ```
   python scripts/train.py --model yolo  # or faster_rcnn
   ```

3. Run real-time detection on RTMP stream:
   ```
   python scripts/detect_rtmp.py --model yolo --rtmp-url rtmp://example.com/live/stream
   ```

## License

[MIT License](LICENSE)

## Acknowledgments

- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [Detectron2](https://github.com/facebookresearch/detectron2)



```
docker run -d \
  --name nginx-rtmp \
  -p 1935:1935 \
  -p 8080:80 \
  alfg/nginx-rtmp
```