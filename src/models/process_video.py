#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.config import YOLOConfig

def setup_argparse():
    parser = argparse.ArgumentParser(description='Process video with YOLOv8 model')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv8 model weights (.pt file)')
    parser.add_argument('--config', type=str, default='configs/visdrone_custom.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections')
    parser.add_argument('--output', type=str, default='output/processed_video.mp4', 
                        help='Path to output video file')
    parser.add_argument('--show', action='store_true', help='Show video while processing')
    parser.add_argument('--device', type=str, default='0', help='Device to run on (e.g., 0 or cpu)')
    parser.add_argument('--track', action='store_true', help='Enable object tracking')
    return parser.parse_args()

def process_video(args):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLOv8 model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # Load config
    print(f"Loading configuration from {args.config}...")
    config = YOLOConfig(args.config)
    class_names = config.config.get('names', {})
    
    # Open the video file
    print(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Process the video
    frame_count = 0
    processing_start = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            # Run detection on the frame
            if args.track:
                results = model.track(frame, conf=args.conf, persist=True, verbose=False)
            else:
                results = model(frame, conf=args.conf, verbose=False)
            
            # Get the annotated frame
            annotated_frame = results[0].plot()
            
            # Write the processed frame to output video
            out.write(annotated_frame)
            
            # Show the frame (if requested)
            if args.show:
                cv2.imshow("YOLOv8 Detection", annotated_frame)
                
                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    
    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        # Calculate processing statistics
        processing_time = time.time() - processing_start
        frames_processed = frame_count
        fps_processing = frames_processed / processing_time if processing_time > 0 else 0
        
        print(f"Processed {frames_processed} frames in {processing_time:.2f} seconds")
        print(f"Processing speed: {fps_processing:.2f} FPS")
        
        # Release resources
        cap.release()
        out.release()
        if args.show:
            cv2.destroyAllWindows()
        
        if frames_processed > 0:
            print(f"Video processing completed. Output saved to: {args.output}")
        else:
            print("No frames were processed.")

if __name__ == "__main__":
    args = setup_argparse()
    process_video(args) 