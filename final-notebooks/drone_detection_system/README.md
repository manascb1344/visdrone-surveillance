# Drone Detection System

This project implements a drone-based object detection system using yolov11 trained on the VisDrone dataset. The system can process drone footage either from recorded files or RTMP streams, detecting objects like people, vehicles, and bicycles.

## Features

- Downloads and processes the VisDrone dataset
- Converts annotations to YOLO format
- Trains yolov11 models with multi-GPU support
- Implements object tracking using yolov11's built-in tracker (BoT-SORT/ByteTrack) (Single Model Mode)
- Implements NMS-based ensembling for detection using multiple trained models (Ensemble Mode)
- Visualizes training results and model performance
- Processes RTMP streams for real-time object detection with tracking (Single Model Mode) or ensembled detections (Ensemble Mode)
- Generates a standalone RTMP processing script for local use with tracking (Single Model Mode Only)

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

The system can be used in different modes via the `main.py` script. Here are examples for each mode:

### 1. Prepare Mode

Downloads the VisDrone dataset and converts annotations to YOLO format. Run this first if you don't have the dataset prepared.

```bash
# Prepare dataset in the default ./data directory
python main.py --mode prepare --base-path ./data 
```

### 2. Train Mode

Trains a YOLOv11 model on the prepared VisDrone dataset.

```bash
# Train a small model (yolov11n) for 50 epochs using 1 GPU
python main.py --mode train --model-size n --epochs 50 --num-gpus 1 --base-path ./data --project-dir ./train_run_n

# Train a medium model (yolov11m) for 100 epochs using 2 GPUs
python main.py --mode train --model-size m --epochs 100 --batch-size 32 --num-gpus 2 --base-path ./data --project-dir ./train_run_m
```
*Note: Adjust `--batch-size` based on your GPU memory.* 
*The best model weights (`best.pt`) will be saved within the specified `--project-dir`.* 

### 3. Export Mode

Exports a trained `.pt` model (from the training step) to an optimized format like ONNX or TensorRT (engine).

```bash
# Export the trained yolov11n model to ONNX format
python main.py --mode export --model-path ./train_run_n/weights/best.pt --export-format onnx

# Export the trained yolov11m model to TensorRT engine format (requires TensorRT)
python main.py --mode export --model-path ./train_run_m/weights/best.pt --export-format engine
```
*The exported model (e.g., `best.onnx` or `best.engine`) will be saved in the same directory as the input `.pt` file.* 

### 4. Test Mode

Tests a trained model (or an ensemble) on sample images from a directory.

```bash
# Test a single trained model (.pt)
python main.py --mode test --model-path ./train_run_n/weights/best.pt --sample-dir ./data/visdrone/VisDrone2019-DET-val/images

# Test a single exported model (.onnx)
python main.py --mode test --model-path ./train_run_n/weights/best.onnx --sample-dir ./data/visdrone/VisDrone2019-DET-val/images

# Test an ensemble of models using NMS (.pt or exported formats)
# (Assumes you copied/renamed best models to a common location like ./models)
python main.py --mode test --model-path ./models/visdrone_yolo_n.pt ./models/visdrone_yolo_m.pt --sample-dir ./data/visdrone/VisDrone2019-DET-val/images
```

### 5. RTMP Mode

Processes a live RTMP video stream using a single model (with tracking) or an ensemble (NMS-based detection).

```bash
# Process with a single model - tracking enabled
python main.py --mode rtmp --model-path ./train_run_n/weights/best.onnx --rtmp-url rtmp://YOUR_STREAM_URL --output-video ./output_n.mp4

# Process with a single model AND generate follow commands for track ID 5
python main.py --mode rtmp --model-path ./train_run_n/weights/best.onnx --rtmp-url rtmp://YOUR_STREAM_URL --follow-id 5

# Process with an ensemble of models - NMS detection, no tracking IDs, no following
python main.py --mode rtmp --model-path ./train_run_n/weights/best.pt ./train_run_m/weights/best.pt --rtmp-url rtmp://YOUR_STREAM_URL --output-video ./output_ensemble.mp4
```
*Replace `rtmp://YOUR_STREAM_URL` with your actual stream URL.* 
*Client script generation (`rtmp_processor.py`) only occurs when running with a single model.* 
*Target following (`--follow-id`) only works with a single model (tracking required).* 

## Example Workflow

*(This section provides a complete sequence, but individual mode examples are above)*

```bash
# Step 1: Prepare the dataset
python main.py --mode prepare --base-path ./data

# Step 2: Train multiple models (e.g., n and s sizes)
python main.py --mode train --model-size n --epochs 50 --batch-size 32 --num-gpus 1 --base-path ./data --project-dir ./train_run_n
python main.py --mode train --model-size s --epochs 50 --batch-size 16 --num-gpus 1 --base-path ./data --project-dir ./train_run_s
# (Optionally copy best.pt from ./train_run_n/weights and ./train_run_s/weights to a common ./models directory)

# Step 3: (Optional) Export models if needed
python main.py --mode export --model-path ./train_run_n/weights/best.pt --export-format onnx
python main.py --mode export --model-path ./train_run_s/weights/best.pt --export-format onnx

# Step 4: Test the ensemble
python main.py --mode test --model-path ./train_run_n/weights/best.onnx ./train_run_s/weights/best.onnx --base-path ./data

# Step 5: Process RTMP feed using the ensemble
python main.py --mode rtmp --model-path ./train_run_n/weights/best.onnx ./train_run_s/weights/best.onnx --rtmp-url rtmp://192.168.1.1/live/stream
```

## Ensembling Note

When providing multiple models via `--model-path` to the `test` or `rtmp` modes, the system performs ensembling by:
1. Running inference with each model independently.
2. Collecting all detected bounding boxes, scores, and classes.
3. Applying Non-Maximum Suppression (NMS) across all collected boxes (per class) to remove redundant detections.

**Important:** This simple NMS ensembling approach does **not** preserve tracking information. The output will show final bounding boxes without individual track IDs when running in ensemble mode.

## Local RTMP Processing

When you run the system in RTMP mode **with a single model**, it generates a standalone script called `rtmp_processor.py` that can be used on a local machine to process RTMP streams with object tracking. **Script generation is disabled when using ensemble mode.**

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics yolov11
- onnx, onnxruntime (for ONNX export/inference)
- CUDA-capable GPU for training (recommended)
- TensorRT (optional, for TensorRT export/inference)

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