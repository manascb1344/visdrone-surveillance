# Drone Detection System

This project implements a drone-based object detection system using YOLO (You Only Look Once) models trained on the VisDrone dataset. The system can process drone footage either from recorded files or RTMP streams, detecting objects like people, vehicles, and bicycles.

## Features

- Downloads and prepares VisDrone dataset for training
- Trains YOLO models in various sizes (nano, small, medium, large, xlarge)
- Processes drone footage from RTMP streams or video files
- Provides visualization of detection results
- Supports object tracking for following specific targets
- Uses efficient batching and CUDA acceleration when available

## Dependencies

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
- NumPy
- Matplotlib

## Project Structure

```
drone_detection_system/
├── main.py             # Main entry point for all operations
├── data_manager.py     # Downloads and prepares the VisDrone dataset
├── model_trainer.py    # Trains the YOLO model
├── visualization.py    # Visualizes training results and model predictions
├── rtmp_processor.py   # Processes RTMP streams with trained models
├── data/               # Dataset storage directory
│   └── visdrone/       # VisDrone dataset
│       ├── VisDrone2019-DET-train/
│       ├── VisDrone2019-DET-val/
│       └── VisDrone2019-DET-test/
└── train_runs/         # Training output directory
    └── yolo/           # YOLO training output directory
        ├── weights/    # Trained model weights
        └── models/     # Exported models for deployment
```

## Usage

### Preparing the Dataset

Downloads and converts the VisDrone dataset into YOLO format:

```bash
python main.py --mode prepare --base-path ./data
```

### Training Models

Trains a YOLO model on the prepared VisDrone dataset.

```bash
# Train a YOLOv8-nano model for 100 epochs using 1 GPU
python main.py --mode train --model-type yolo --model-size n --epochs 100 --batch-size 16 --num-gpus 1 --base-path ./data --project-dir ./train_runs
```

### Testing Trained Models

Test a trained model on validation images:

```bash
# Test a YOLO model
python main.py --mode test --model-path ./train_runs/yolo/weights/best.pt --sample-dir ./data/visdrone/VisDrone2019-DET-val/images
```

### Processing RTMP Streams

Process a live RTMP stream using a trained model:

```bash
# Process with a single YOLO model with tracking
python main.py --mode rtmp --model-path ./train_runs/yolo/weights/best.pt --rtmp-url rtmp://YOUR_STREAM_URL --output-video ./output_yolo.mp4
```

#### Target Following

Enable following a specific detected object by its tracking ID:

```bash
# Follow object with tracking ID 2
python main.py --mode rtmp --model-path ./train_runs/yolo/weights/best.pt --rtmp-url rtmp://YOUR_STREAM_URL --follow-id 2
```

#### Real-time Display

Enable real-time visualization of the detection results:

```bash
# Process with real-time display
python main.py --mode rtmp --model-path ./train_runs/yolo/weights/best.pt --rtmp-url rtmp://YOUR_STREAM_URL --realtime-display
```

### Exporting Models

Export a trained PyTorch model to ONNX format for faster inference:

```bash
python main.py --mode export --model-path ./train_runs/yolo/weights/best.pt --export-format onnx
```

## Example Workflow

Here's a complete workflow example:

```bash
# 1. Prepare the dataset
python main.py --mode prepare --base-path ./data

# 2. Train a model
python main.py --mode train --model-type yolo --epochs 100 --batch-size 16 --num-gpus 1 --base-path ./data --project-dir ./train_runs

# 3. Test the trained model
python main.py --mode test --model-path ./train_runs/yolo/weights/best.pt --base-path ./data

# 4. Process an RTMP stream
python main.py --mode rtmp --model-path ./train_runs/yolo/weights/best.onnx --rtmp-url rtmp://192.168.1.1/live/stream
```

## Notes

- For optimal performance, ensure your GPU has sufficient VRAM for the selected model size
- Larger models (l, x) require more memory but generally provide better detection accuracy
- The system automatically uses CUDA if available
- Multiple models of the same type can be used in ensemble mode for potentially improved results

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