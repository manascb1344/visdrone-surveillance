# Drone Detection System

This project implements a drone-based object detection system using yolov11 and Faster R-CNN models trained on the VisDrone dataset. The system can process drone footage either from recorded files or RTMP streams, detecting objects like people, vehicles, and bicycles.

## Features

- Downloads and prepares the VisDrone dataset for object detection
- Trains yolov11 models with multi-GPU support
- Trains Faster R-CNN models on the same dataset format
- Implements NMS-based ensembling for detection using multiple trained models (Ensemble Mode)
- Visualizes training results and model performance
- Processes RTMP streams for real-time object detection with tracking (Single Model Mode) or ensembled detections (Ensemble Mode)
- Generates drone movement commands for "following" specific objects (Single YOLO Model Mode only)
- Exports models to optimized formats (ONNX, TensorRT)

## Requirements

- Python 3.8+
- PyTorch 1.10+
- OpenCV
- Ultralytics (for YOLO models)
- TorchVision (for Faster R-CNN models)
- CUDA-capable GPU (recommended for training)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/drone-detection-system.git
cd drone-detection-system

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
├── main.py               # Main script for running operations
├── data_manager.py       # Prepares and converts the VisDrone dataset
├── model_trainer.py      # Trains the YOLO model
├── faster_rcnn.py        # Trains the Faster R-CNN model
├── visualization.py      # Visualizes results and model performance
├── rtmp_processor.py     # Processes RTMP streams for real-time detection
├── visdrone_yolo/        # Converted dataset directory
│   ├── dataset.yaml      # Dataset configuration
│   ├── images/           # Training and validation images
│   └── labels/           # YOLO format labels
└── visdrone_yolo_training/  # Training output directory
    └── (training runs)
└── visdrone_faster_rcnn_training/  # Faster R-CNN training output directory
    └── (training runs)
```

## Usage

### 1. Prepare Mode

Downloads and prepares the VisDrone dataset:

```bash
python main.py --mode prepare --base-path ./data
```

### 2. Train Mode

Trains a model (YOLO or Faster R-CNN) on the prepared VisDrone dataset.

#### Training YOLO models:

```bash
# Train a small YOLO model (yolov11n) for 50 epochs using 1 GPU
python main.py --mode train --model-type yolo --model-size n --epochs 50 --num-gpus 1 --base-path ./data --project-dir ./train_runs

# Train a medium YOLO model (yolov11m) for 100 epochs using 2 GPUs
python main.py --mode train --model-type yolo --model-size m --epochs 100 --batch-size 32 --num-gpus 2 --base-path ./data --project-dir ./train_runs
```

#### Training Faster R-CNN models:

```bash
# Train a Faster R-CNN model for 50 epochs using 1 GPU
python main.py --mode train --model-type faster_rcnn --epochs 50 --batch-size 8 --num-gpus 1 --base-path ./data --project-dir ./train_runs --learning-rate 0.001
```

### 3. Export Mode

Exports a trained `.pt` model (from the YOLO training step) to an optimized format like ONNX or TensorRT (engine). Note: Export is currently only supported for YOLO models.

```bash
# Export the trained yolov11n model to ONNX format
python main.py --mode export --model-path ./train_runs/yolo/weights/best.pt --export-format onnx

# Export the trained yolov11m model to TensorRT engine format (requires TensorRT)
python main.py --mode export --model-path ./train_runs/yolo/weights/best.pt --export-format engine
```

### 4. Test Mode

Tests a trained model (or an ensemble) on sample images from a directory.

```bash
# Test a single YOLO model (.pt)
python main.py --mode test --model-path ./train_runs/yolo/weights/best.pt --sample-dir ./data/visdrone/VisDrone2019-DET-val/images

# Test an exported YOLO model (.onnx)
python main.py --mode test --model-path ./train_runs/yolo/weights/best.onnx --sample-dir ./data/visdrone/VisDrone2019-DET-val/images

# Test a Faster R-CNN model
python main.py --mode test --model-path ./train_runs/faster_rcnn/weights/best.pt --sample-dir ./data/visdrone/VisDrone2019-DET-val/images

# Test an ensemble of models using NMS (.pt or exported formats)
python main.py --mode test --model-path ./train_runs/yolo/weights/best.pt ./train_runs/faster_rcnn/weights/best.pt --sample-dir ./data/visdrone/VisDrone2019-DET-val/images
```

### 5. RTMP Mode

Processes a live RTMP video stream using a single model (with tracking) or an ensemble (NMS-based detection).

```bash
# Process with a single YOLO model - enables tracking IDs
python main.py --mode rtmp --model-path ./train_runs/yolo/weights/best.onnx --rtmp-url rtmp://YOUR_STREAM_URL --output-video ./output_n.mp4

# Process with a single YOLO model - tracking with "follow" commands for a specific track ID
python main.py --mode rtmp --model-path ./train_runs/yolo/weights/best.onnx --rtmp-url rtmp://YOUR_STREAM_URL --follow-id 5

# Process with a single YOLO model - real-time display of processed video feed
python main.py --mode rtmp --model-path ./train_runs/yolo/weights/best.onnx --rtmp-url rtmp://YOUR_STREAM_URL --realtime-display

# Process with a single YOLO model - real-time display with frame skipping for better performance
python main.py --mode rtmp --model-path ./train_runs/yolo/weights/best.onnx --rtmp-url rtmp://YOUR_STREAM_URL --realtime-display --max-frame-skip 10

# Process with a single Faster R-CNN model - no tracking IDs
python main.py --mode rtmp --model-path ./train_runs/faster_rcnn/weights/best.pt --rtmp-url rtmp://YOUR_STREAM_URL --output-video ./output_rcnn.mp4

# Process with an ensemble of models - NMS detection, no tracking IDs, no following
python main.py --mode rtmp --model-path ./train_runs/yolo/weights/best.pt ./train_runs/faster_rcnn/weights/best.pt --rtmp-url rtmp://YOUR_STREAM_URL --output-video ./output_ensemble.mp4
```

You can combine options like `--realtime-display` with `--output-video` to simultaneously view and save the processed feed.

## Example Workflow

Here's a complete workflow example:

```bash
# Step 1: Prepare the dataset
python main.py --mode prepare --base-path ./data

# Step 2: Train multiple models
python main.py --mode train --model-type yolo --model-size n --epochs 50 --batch-size 32 --num-gpus 1 --base-path ./data --project-dir ./train_runs
python main.py --mode train --model-type faster_rcnn --epochs 50 --batch-size 8 --num-gpus 1 --base-path ./data --project-dir ./train_runs
# (Optionally copy best.pt from ./train_runs/yolo/weights and ./train_runs/faster_rcnn/weights to a common ./models directory)

# Step 3: Export YOLO model to ONNX
python main.py --mode export --model-path ./train_runs/yolo/weights/best.pt --export-format onnx

# Step 4: Test the ensemble
python main.py --mode test --model-path ./train_runs/yolo/weights/best.onnx ./train_runs/faster_rcnn/weights/best.pt --base-path ./data

# Step 5: Process RTMP feed using the ensemble
python main.py --mode rtmp --model-path ./train_runs/yolo/weights/best.onnx ./train_runs/faster_rcnn/weights/best.pt --rtmp-url rtmp://192.168.1.1/live/stream
```

## Notes

- The dataset is automatically downloaded the first time you run the prepare or train modes.
- Training with multiple GPUs requires PyTorch with CUDA support.
- The "follow" feature generates basic movement commands that could be used to control a drone to follow a specific object. This feature outputs text that would need to be integrated with actual drone control systems.
- The follow feature only works when using a single YOLO model. It cannot generate individual track IDs when running in ensemble mode.
- **Object tracking is only available with YOLO models. Faster R-CNN models do not support object tracking in this implementation.**
- Hybrid Ensemble Mode allows combining both YOLO and Faster R-CNN models for potentially improved detection results.

## System Requirements

- OS: Windows, Mac, or Linux
- Python 3.8+
- PyTorch 1.10+ (with CUDA for GPU acceleration)
- TorchVision 0.11+
- 8GB+ RAM
- CUDA-capable GPU for training (recommended)

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
- Ultralytics yolov11
- onnx, onnxruntime (for ONNX export/inference)
- CUDA-capable GPU for training (recommended)
- TensorRT (optional, for TensorRT export/inference) 