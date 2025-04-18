#!/usr/bin/env python3
import argparse
import os
import cv2
import numpy as np

def setup_argparse():
    parser = argparse.ArgumentParser(description='Extract frames from video')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='output/frames', help='Output directory for frames')
    parser.add_argument('--frames', type=str, default='0.25,0.5,0.75', 
                        help='Comma-separated list of frame positions (as percentages)')
    parser.add_argument('--format', type=str, default='jpg', help='Output format (jpg or png)')
    return parser.parse_args()

def extract_frames(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Open the video file
    print(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video properties: {total_frames} frames, {fps} FPS, {duration:.2f} seconds")
    
    # Parse the frame positions
    try:
        percentages = [float(p) for p in args.frames.split(',')]
        frame_positions = [int(p * total_frames) for p in percentages]
    except ValueError:
        print(f"Error: Invalid frame positions format. Use comma-separated percentages (e.g., '0.25,0.5,0.75')")
        return
    
    # Extract the frames
    frames_extracted = 0
    
    for i, frame_pos in enumerate(frame_positions):
        # Ensure frame position is within bounds
        frame_pos = max(0, min(frame_pos, total_frames - 1))
        
        # Set the video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame at position {frame_pos}")
            continue
        
        # Save the frame
        output_file = os.path.join(args.output, f"frame_{i+1}_{percentages[i]:.2f}.{args.format}")
        cv2.imwrite(output_file, frame)
        frames_extracted += 1
        
        print(f"Extracted frame at position {frame_pos}/{total_frames} ({percentages[i]:.2%})")
    
    # Release resources
    cap.release()
    
    print(f"Extracted {frames_extracted} frames to {args.output}")

if __name__ == "__main__":
    args = setup_argparse()
    extract_frames(args) 