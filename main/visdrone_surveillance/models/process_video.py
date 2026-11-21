#!/usr/bin/env python3
import argparse
import os
import sys
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO

from .yolo_config import YOLOConfig

def setup_argparse():
    parser = argparse.ArgumentParser(description='Process video with YOLOv8 model for real-time detection')
    
    # Input source options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--video', type=str, help='Path to input video file')
    input_group.add_argument('--webcam', type=int, default=None, help='Webcam device number (e.g., 0 for default camera)')
    input_group.add_argument('--rtsp', type=str, help='RTSP stream URL')
    
    # Model and inference options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--model', type=str, required=True, help='Path to YOLOv8 model weights (.pt file)')
    model_group.add_argument('--config', type=str, default='configs/visdrone_custom.yaml', 
                        help='Path to configuration file')
    model_group.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections')
    model_group.add_argument('--device', type=str, default='0', help='Device to run on (e.g., 0 or cpu)')
    model_group.add_argument('--half', action='store_true', help='Use half precision (FP16) for faster inference')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', type=str, default='output/processed_video.mp4', 
                        help='Path to output video file')
    output_group.add_argument('--show', action='store_true', help='Show video while processing')
    output_group.add_argument('--stream', action='store_true', help='Stream output as MJPEG HTTP server')
    output_group.add_argument('--stream-port', type=int, default=8080, help='Port for HTTP streaming server')
    output_group.add_argument('--no-save', action='store_true', help='Do not save output video')
    
    # Detection and tracking options
    detect_group = parser.add_argument_group('Detection Options')
    detect_group.add_argument('--track', action='store_true', default=True, help='Enable object tracking')
    detect_group.add_argument('--count-objects', action='store_true', help='Count detected objects per class')
    detect_group.add_argument('--trails', action='store_true', help='Show object tracking trails')
    detect_group.add_argument('--trail-length', type=int, default=30, help='Length of tracking trails in frames')
    
    # Performance optimization options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument('--resize', type=int, default=None, help='Resize input width for faster processing (height adjusted automatically)')
    perf_group.add_argument('--skip-frames', type=int, default=0, help='Skip N frames between detections for faster processing')
    perf_group.add_argument('--max-fps', type=int, default=0, help='Limit maximum FPS (0 for no limit)')
    
    return parser.parse_args()

class ObjectCounter:
    def __init__(self, class_names):
        self.class_names = class_names
        self.counts = defaultdict(int)
        self.frame_counts = defaultdict(int)
        self.tracked_objects = set()
        
    def update(self, results, frame_idx):
        """Update counts based on detection results"""
        self.frame_counts = defaultdict(int)
        
        if not hasattr(results[0], 'boxes'):
            return
        
        # If tracking is enabled
        if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            for box, cls, track_id in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.id):
                cls_name = self.class_names[int(cls)]
                track_id = int(track_id)
                obj_id = f"{cls_name}_{track_id}"
                
                # Count unique objects
                if obj_id not in self.tracked_objects:
                    self.tracked_objects.add(obj_id)
                    self.counts[cls_name] += 1
                
                # Count per frame
                self.frame_counts[cls_name] += 1
        # Without tracking, just count all objects in the frame
        else:
            for cls in results[0].boxes.cls:
                cls_name = self.class_names[int(cls)]
                self.counts[cls_name] += 1
                self.frame_counts[cls_name] += 1
    
    def draw_counts(self, frame):
        """Draw count information on the frame"""
        # Add semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 35 + len(self.counts) * 25), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Add title
        cv2.putText(frame, "Object Counts (Current Frame | Total)", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add counts for each class
        y_pos = 60
        for i, (cls_name, count) in enumerate(sorted(self.counts.items())):
            frame_count = self.frame_counts[cls_name]
            text = f"{cls_name}: {frame_count} | {count}"
            cv2.putText(frame, text, (15, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25
        
        return frame

class TrailTracker:
    def __init__(self, max_trail_length=30):
        self.max_trail_length = max_trail_length
        self.trails = defaultdict(lambda: deque(maxlen=max_trail_length))
        self.colors = {}
    
    def update(self, results):
        """Update object trails based on detection results"""
        if not hasattr(results[0], 'boxes') or not hasattr(results[0].boxes, 'id') or results[0].boxes.id is None:
            return
        
        for box, cls, track_id in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.id):
            # Convert box to center point (x, y)
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Generate unique ID and color for this object
            obj_id = f"{int(cls)}_{int(track_id)}"
            if obj_id not in self.colors:
                self.colors[obj_id] = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Add center point to the trail
            self.trails[obj_id].append(center)
    
    def draw_trails(self, frame):
        """Draw object trails on the frame"""
        for obj_id, trail in self.trails.items():
            # Skip if trail is too short
            if len(trail) < 2:
                continue
            
            # Draw polylines of the trail
            points = np.array(list(trail), dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], False, self.colors[obj_id], thickness=2)
        
        return frame

class VideoStreamer:
    def __init__(self, port=8080):
        self.port = port
        self.server_started = False
        
    def start_server(self):
        try:
            import threading
            from http.server import HTTPServer, BaseHTTPRequestHandler
            
            class StreamingHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write("""
                        <html>
                        <head>
                        <title>YOLOv8 Drone Detection Stream</title>
                        </head>
                        <body>
                        <h1>YOLOv8 Drone Detection Stream</h1>
                        <img src="/stream"/>
                        </body>
                        </html>
                        """.encode())
                    elif self.path == '/stream':
                        self.send_response(200)
                        self.send_header('Age', '0')
                        self.send_header('Cache-Control', 'no-cache, private')
                        self.send_header('Pragma', 'no-cache')
                        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
                        self.end_headers()
                        try:
                            while True:
                                if hasattr(self.server, 'frame_jpeg') and self.server.frame_jpeg is not None:
                                    self.wfile.write(b'--FRAME\r\n')
                                    self.send_header('Content-Type', 'image/jpeg')
                                    self.send_header('Content-Length', len(self.server.frame_jpeg))
                                    self.end_headers()
                                    self.wfile.write(self.server.frame_jpeg)
                                    self.wfile.write(b'\r\n')
                                time.sleep(0.05)  # 20 fps
                        except Exception as e:
                            print(f"Stream connection ended: {e}")
                    else:
                        self.send_error(404)
                        self.end_headers()
            
            class StreamingServer(HTTPServer):
                def __init__(self, *args, **kwargs):
                    self.frame_jpeg = None
                    super().__init__(*args, **kwargs)
            
            # Start server in a new thread
            print(f"Starting streaming server on port {self.port}")
            self.server = StreamingServer(('', self.port), StreamingHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.server_started = True
            print(f"Streaming server started at http://localhost:{self.port}")
            
        except Exception as e:
            print(f"Failed to start streaming server: {e}")
            self.server_started = False
    
    def update_frame(self, frame):
        """Update the current frame for streaming"""
        if not self.server_started:
            return
        
        # Convert to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            self.server.frame_jpeg = jpeg.tobytes()

class FPSMonitor:
    def __init__(self, avg_frames=30):
        self.prev_time = time.time()
        self.curr_fps = 0
        self.avg_frames = avg_frames
        self.fps_history = deque(maxlen=avg_frames)
    
    def update(self):
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        self.fps_history.append(fps)
        self.curr_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        return self.curr_fps
    
    def draw(self, frame):
        fps_text = f"FPS: {self.curr_fps:.1f}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 110, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

def process_video(args):
    # Validate input source
    if not args.video and args.webcam is None and not args.rtsp:
        print("Error: You must provide either a video file (--video), webcam device (--webcam), or RTSP stream (--rtsp)")
        return
    
    # Create output directory if saving video
    if not args.no_save:
        output_dir = os.path.dirname(args.output)
        os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLOv8 model
    print(f"Loading model from {args.model}...")
    if args.half:
        print("Using half precision (FP16) for faster inference")
    model = YOLO(args.model)
    
    # Load config
    print(f"Loading configuration from {args.config}...")
    config = YOLOConfig(args.config)
    class_names = config.config.get('names', {})
    
    # Initialize object counter if requested
    counter = None
    if args.count_objects:
        counter = ObjectCounter(class_names)
    
    # Initialize trail tracker if requested
    tracker = None
    if args.trails:
        tracker = TrailTracker(args.trail_length)
    
    # Initialize video streamer if requested
    streamer = None
    if args.stream:
        streamer = VideoStreamer(args.stream_port)
        streamer.start_server()
    
    # Initialize FPS monitor
    fps_monitor = FPSMonitor()
    
    # Open video source
    if args.video:
        print(f"Opening video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
    elif args.webcam is not None:
        print(f"Opening webcam device: {args.webcam}")
        cap = cv2.VideoCapture(args.webcam)
        # Set webcam properties for better performance
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    elif args.rtsp:
        print(f"Opening RTSP stream: {args.rtsp}")
        cap = cv2.VideoCapture(args.rtsp)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source")
        return
    
    # Get video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate aspect ratio and new dimensions if resizing
    if args.resize:
        new_width = args.resize
        new_height = int(orig_height * (new_width / orig_width))
        print(f"Resizing input from {orig_width}x{orig_height} to {new_width}x{new_height}")
    else:
        new_width, new_height = orig_width, orig_height
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.video:  # Only available for video files
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties: {orig_width}x{orig_height}, {fps} FPS, {total_frames} frames")
    else:
        total_frames = float('inf')
        print(f"Stream properties: {orig_width}x{orig_height}, {fps} FPS")
    
    # Initialize the video writer if saving
    out = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps if fps > 0 else 30, (new_width, new_height))
    
    # Process the video
    frame_count = 0
    processing_frames = 0
    skip_count = 0
    processing_start = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if args.video:  # Only break for video files
                    break
                else:  # For webcams and streams, try to reconnect
                    print("Lost connection to camera. Attempting to reconnect...")
                    time.sleep(1)
                    continue
            
            frame_count += 1
            
            # Resize frame if requested
            if args.resize:
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Frame skipping logic for performance
            if args.skip_frames > 0:
                if skip_count < args.skip_frames:
                    skip_count += 1
                    
                    # Still need to draw previous detection results
                    if 'annotated_frame' in locals():
                        if args.show:
                            cv2.imshow("YOLOv8 Detection", annotated_frame)
                        if not args.no_save:
                            out.write(annotated_frame)
                        if streamer:
                            streamer.update_frame(annotated_frame)
                    
                    # Continue to next frame without detection
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                else:
                    skip_count = 0
            
            # Process frame count (only counting frames that actually get processed)
            processing_frames += 1
            if args.video and processing_frames % 10 == 0:
                print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            # FPS limiting logic
            if args.max_fps > 0:
                elapsed = time.time() - processing_start
                target_time = processing_frames / args.max_fps
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
            
            # Run detection/tracking on the frame
            if args.track:
                results = model.track(frame, conf=args.conf, persist=True, verbose=False, half=args.half)
            else:
                results = model(frame, conf=args.conf, verbose=False, half=args.half)
            
            # Update object counter if enabled
            if counter:
                counter.update(results, frame_count)
            
            # Update tracking trails if enabled
            if tracker:
                tracker.update(results)
            
            # Get the annotated frame
            annotated_frame = results[0].plot()
            
            # Add object count information if enabled
            if counter:
                annotated_frame = counter.draw_counts(annotated_frame)
            
            # Add tracking trails if enabled
            if tracker:
                annotated_frame = tracker.draw_trails(annotated_frame)
            
            # Update and draw FPS
            current_fps = fps_monitor.update()
            annotated_frame = fps_monitor.draw(annotated_frame)
            
            # Add timestamp for video files
            if args.video:
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                            (new_width - 250, new_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 255, 255), 2)
            
            # Update streaming frame if enabled
            if streamer:
                streamer.update_frame(annotated_frame)
            
            # Write the processed frame to output video if saving
            if not args.no_save:
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
        import traceback
        traceback.print_exc()
    
    finally:
        # Calculate processing statistics
        processing_time = time.time() - processing_start
        frames_processed = processing_frames
        total_frames_received = frame_count
        fps_processing = frames_processed / processing_time if processing_time > 0 else 0
        
        print(f"Processed {frames_processed} frames (received {total_frames_received}) in {processing_time:.2f} seconds")
        print(f"Processing speed: {fps_processing:.2f} FPS")
        
        # Print object counts if enabled
        if counter:
            print("\nTotal object counts:")
            for cls_name, count in sorted(counter.counts.items()):
                print(f"  {cls_name}: {count}")
        
        # Release resources
        cap.release()
        if out:
            out.release()
        if args.show:
            cv2.destroyAllWindows()
        
        if frames_processed > 0:
            if not args.no_save:
                print(f"Video processing completed. Output saved to: {args.output}")
            if args.stream:
                print(f"Stream available at http://localhost:{args.stream_port}")
        else:
            print("No frames were processed.")

if __name__ == "__main__":
    args = setup_argparse()
    process_video(args) 