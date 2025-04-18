# Product Requirements Document: Real-Time Drone Object Detection System (VisDrone-RTMP)

**Version:** 1.0
**Date:** October 26, 2023

## 1. Introduction

This document outlines the requirements for a system capable of performing real-time object detection on live video feeds from drones, specifically focusing on identifying and tracking vehicles and people. The system will utilize the VisDrone dataset for training and leverage state-of-the-art machine learning models.

The system aims to process a live video feed from a drone, received via the Real-Time Messaging Protocol (RTMP), and perform real-time object detection to identify and locate specific classes of objects. Two distinct object detection models will be developed and trained: one based on a single-stage detector architecture (like YOLO) and another based on a two-stage detector architecture (like Faster R-CNN).

### 1.1. Goals

* Develop and train two separate object detection models (YOLO-based and Faster R-CNN-based) specifically tailored for aerial drone imagery using the VisDrone dataset.
* Implement a robust mechanism to ingest and decode a live RTMP video stream from a drone.
* Perform real-time object detection inference (targeting vehicles and people initially) on the incoming RTMP video frames using one of the trained models (user-selectable at runtime).
* Visualize the detection results (bounding boxes, class labels) by overlaying them onto the video feed in real-time.
* Provide a baseline system demonstrating the feasibility of real-time monitoring using drone feeds and state-of-the-art object detection techniques.
* Evaluate and compare the performance (accuracy vs. speed) of the two different model architectures in this specific application context.

### 1.2. Scope (Version 1.0)

#### In Scope:
* **Data Handling:**
  * Downloading and preprocessing the VisDrone dataset (specifically the object detection subset).
  * Converting VisDrone annotations to formats suitable for the chosen model training frameworks.
  * Data augmentation techniques suitable for aerial imagery (e.g., rotation, scaling, brightness/contrast adjustment).
* **Model Development & Training:**
  * Implementation and training of a YOLO-family model (e.g., YOLOv8, YOLOv9) on VisDrone data.
  * Implementation and training of a Faster R-CNN family model on VisDrone data.
  * Focus on detecting predefined classes relevant to VisDrone, with initial emphasis on 'pedestrian', 'person', 'car', 'van', 'bus', 'truck'.
  * Model evaluation using standard metrics (e.g., mAP) on a VisDrone validation set.
  * Saving trained model weights for inference.
* **Inference Engine:**
  * Ability to load pre-trained model weights (either YOLO-like or Faster R-CNN-like).
  * Ingestion of a single RTMP video stream.
  * Frame decoding and preprocessing for model input.
  * Running object detection inference on each frame (or selectively, based on performance).
  * Post-processing of model outputs (e.g., non-max suppression).
* **Visualization:**
  * Drawing bounding boxes around detected objects on the video frames.
  * Displaying class labels (and optionally confidence scores) near the bounding boxes.
  * Displaying the annotated video feed in a real-time window or outputting it to another stream/file.
  * Display the real-time processing speed (Frames Per Second - FPS) on the output feed or console.

#### Out of Scope (Future Considerations):
* **Object Segmentation:** While initially considered, segmentation mask prediction and plotting are out of scope for V1 due to significantly higher computational cost impacting real-time performance.
* **Object Tracking:** Tracking objects across multiple frames is not included. Each frame is processed independently for detection.
* **Advanced Analytics:** No complex analytics like object counting trends, speed estimation, trajectory analysis, or anomaly detection.
* **User Interface (UI):** No graphical user interface (GUI) beyond the basic video output window. Control will be via command-line or configuration files.
* **Multi-Stream Handling:** The system will only process one RTMP stream at a time.
* **Edge Deployment:** Inference is assumed to run on a server/workstation with sufficient compute resources (GPU recommended).
* **Model Quantization/Optimization:** Advanced model optimization techniques for speedup are not included initially.
* **Re-streaming Output:** While displaying the output is in scope, re-streaming the annotated video feed via RTMP/RTSP is out of scope for V1.
* **Authentication/Security:** No specific security measures for the RTMP stream input.

## 2. Functional Requirements

### 2.1. Data Handling
* **FR-1:** System must ingest and process the VisDrone dataset (images and annotations) for training and validation.
* **FR-2:** Implement data augmentation techniques suitable for aerial imagery (e.g., rotation, scaling, brightness/contrast adjustment).
* **FR-3:** Support splitting the dataset into training, validation, and test sets.

### 2.2. Model Training
* **FR-4:** Implement training pipeline for the YOLO-based model using the processed VisDrone data.
* **FR-5:** Implement training pipeline for the Faster R-CNN-based model using the processed VisDrone data.
* **FR-6:** Training process should support configurable hyperparameters (learning rate, batch size, epochs, optimizer).
* **FR-7:** Training should utilize GPU acceleration (e.g., CUDA).
* **FR-8:** Save trained model checkpoints and associated training logs (loss, accuracy metrics).

### 2.3. Real-Time Inference
* **FR-9:** System must accept a live RTMP video stream URL as input.
* **FR-10:** Decode and preprocess incoming video frames for model inference.
* **FR-11:** Perform inference using either the trained YOLO-based or Faster R-CNN-based model (selectable at runtime or via configuration).
* **FR-12:** Process model outputs (bounding boxes, confidence scores) for 'vehicle' and 'people' classes.
* **FR-13:** Filter detections based on a configurable confidence threshold.

### 2.4. Visualization
* **FR-14:** Overlay bounding boxes (with class labels and confidence scores) onto the corresponding objects in the video frames.
* **FR-15:** Output the annotated video stream (potentially via another RTMP stream, file, or display window).
* **FR-16:** Display the real-time processing speed (Frames Per Second - FPS) on the output feed or console.

## 3. Non-Functional Requirements

* **NFR-1: Performance:** The inference pipeline (frame capture, preprocessing, inference, postprocessing, visualization) should aim for near real-time performance (e.g., >10-15 FPS) on suitable hardware, acknowledging the computational cost difference between the models.
* **NFR-2: Accuracy:** Models should achieve reasonable accuracy on the VisDrone validation set for vehicle and people categories (specific mAP targets to be determined during development based on baseline models, e.g., > 20-25 mAP @ IoU=0.50 for relevant classes).
* **NFR-3: Reliability:** The system should be able to run continuously for extended periods (e.g., > 1 hour) on a stable RTMP stream without crashing. It should handle stream disconnection events gracefully.
* **NFR-4: Usability:** The system should be runnable via command-line interface, accepting necessary configurations (RTMP URL, model choice, confidence threshold).
* **NFR-5: Modularity:** Code should be organized into logical modules (data handling, model definition, training, inference, visualization) to facilitate maintenance and future extensions.
* **NFR-6: Technology Stack:** Primarily Python. Key libraries: PyTorch or TensorFlow/Keras, OpenCV, potentially libraries for RTMP handling (e.g., ffmpeg-python, GStreamer bindings) and model implementations (e.g., Detectron2, Ultralytics YOLO).
* **NFR-7: Resource Consumption:** The system should operate within the memory limits of the target hardware (e.g., < 8GB GPU VRAM during inference). CPU usage should also be monitored.

## 4. Data Requirements

* **DR-1:** Access to the VisDrone dataset (specifically VisDrone-DET challenge data). [https://github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)
* **DR-2:** Annotations should be parsed correctly corresponding to image files. Focus on 'pedestrian', 'person', 'car', 'van', 'bus', 'truck' categories, potentially grouping them into 'people' and 'vehicles'.
* **DR-3:** Input Video Stream: Live RTMP feed from a drone. Assumed characteristics: H.264 encoded video, resolutions like 720p or 1080p, variable frame rate.

## 5. System Architecture (High-Level)
--- 