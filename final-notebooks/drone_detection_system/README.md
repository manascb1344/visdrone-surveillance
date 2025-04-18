# Drone Detection System

This project implements a drone-based object detection system using YOLOv8 trained on the VisDrone dataset. The system can process drone footage either from recorded files or RTMP streams, detecting objects like people, vehicles, and bicycles.

## Features

- Downloads and processes the VisDrone dataset
- Converts annotations to YOLO format
- Trains YOLOv8 models with multi-GPU support
- Implements object tracking using YOLOv8's built-in tracker (BoT-SORT/ByteTrack)
- Visualizes training results and model performance
- Processes RTMP streams for real-time object detection with tracking
- Generates a standalone RTMP processing script for local use with tracking

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt # Installs core libraries + ONNX Runtime (CPU)
# For GPU acceleration with ONNX, install:
# pip install onnxruntime-gpu 
# For TensorRT (NVIDIA GPU required), install TensorRT and:
# pip install tensorrt
```

## Project Structure

```
drone_detection_system/
├── main.py                # Main script to run the system
├── data_manager.py        # Handles dataset operations
├── model_trainer.py       # Trains the YOLO model
├── visualization.py       # Visualizes results
├── rtmp_processor.py      # Processes RTMP feeds
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Usage

The system can be used in four different modes:

### 1. Prepare Mode

Downloads and prepares the VisDrone dataset:

```bash
python main.py --mode prepare --base-path ./data
```

### 2. Train Mode

Trains a YOLOv8 model on the VisDrone dataset:

```bash
python main.py --mode train --model-size n --epochs 25 --batch-size 64 --num-gpus 1
```

### 3. Test Mode

Tests a trained model on sample images:

```bash
python main.py --mode test --model-path ./visdrone_yolo_training/models/visdrone_yolo_n.pt --sample-dir ./data/visdrone/VisDrone2019-DET-val/images
```

### 4. RTMP Mode

Processes an RTMP feed with a trained model (`.pt`, `.onnx`, `.engine`), displaying tracked objects with unique IDs:

```bash
python main.py --mode rtmp --model-path /path/to/model.onnx --rtmp-url rtmp://server/live/stream --output-video output.mp4
```

### 5. Export Mode (New)

Exports a trained `.pt` model to an optimized format (ONNX or TensorRT engine):

```bash
# Export to ONNX (default)
python main.py --mode export --model-path ./visdrone_yolo_training/models/visdrone_yolo_n.pt --export-format onnx

# Export to TensorRT engine (requires TensorRT installation)
python main.py --mode export --model-path ./visdrone_yolo_training/models/visdrone_yolo_n.pt --export-format engine
```

## Example Workflow

```bash
# Step 1: Prepare the dataset
python main.py --mode prepare --base-path ./data

# Step 2: Train the model
python main.py --mode train --epochs 50 --batch-size 32 --num-gpus 2 --base-path ./data --project-dir ./visdrone_yolo_training

# Step 3: (Optional) Export the model for potentially faster inference
python main.py --mode export --model-path ./visdrone_yolo_training/models/visdrone_yolo_n.pt --export-format onnx

# Step 4: Test the exported model
python main.py --mode test --model-path ./visdrone_yolo_training/models/visdrone_yolo_n.onnx --base-path ./data

# Step 5: Process RTMP feed using the exported model
python main.py --mode rtmp --model-path ./visdrone_yolo_training/models/visdrone_yolo_n.onnx --rtmp-url rtmp://192.168.1.1/live/stream
```

## Class Names

The system detects and tracks the following classes:
- person
- bicycle
- car
- van
- truck
- tricycle
- bus
- motorcycle

## Local RTMP Processing

When you run the system in RTMP mode, it generates a standalone script called `rtmp_processor.py` that can be used on a local machine to process RTMP streams with object tracking. This is useful for deploying the model in the field.

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLOv8
- onnx, onnxruntime (for ONNX export/inference)
- CUDA-capable GPU for training (recommended)
- TensorRT (optional, for TensorRT export/inference) 