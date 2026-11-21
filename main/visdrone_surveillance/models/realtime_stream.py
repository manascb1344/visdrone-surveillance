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
from threading import Thread
from queue import Queue, Full, Empty
from ultralytics import YOLO

from .yolo_config import YOLOConfig

def setup_argparse():
    parser = argparse.ArgumentParser(description='Process RTMP streams in real-time with YOLOv8')
    
    # Input source options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--rtmp', type=str, help='RTMP stream URL', required=True)
    
    # Model and inference options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--model', type=str, required=True, help='Path to YOLOv8 model weights (.pt file)')
    model_group.add_argument('--config', type=str, default='configs/visdrone_custom.yaml', 
                        help='Path to configuration file')
    model_group.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections')
    model_group.add_argument('--device', type=str, default='0', help='Device to run on (e.g., 0 or cpu)')
    model_group.add_argument('--half', action='store_true', default=True, help='Use half precision (FP16) for faster inference')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--stream', action='store_true', default=True, help='Stream output as MJPEG HTTP server')
    output_group.add_argument('--stream-port', type=int, default=8080, help='Port for HTTP streaming server')
    output_group.add_argument('--show', action='store_true', help='Show video while processing')
    
    # Detection and tracking options
    detect_group = parser.add_argument_group('Detection Options')
    detect_group.add_argument('--track', action='store_true', default=True, help='Enable object tracking')
    detect_group.add_argument('--count-objects', action='store_true', default=True, help='Count detected objects per class')
    
    # Performance optimization options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument('--resize', type=int, default=320, help='Resize input width for faster processing (height adjusted automatically)')
    perf_group.add_argument('--skip-frames', type=int, default=4, help='Skip N frames between detections for faster processing')
    perf_group.add_argument('--buffer-size', type=int, default=10, help='Maximum buffer size for frames in the queue')
    perf_group.add_argument('--max-latency', type=float, default=0.5, help='Maximum latency in seconds before dropping frames')
    
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

class FPSMonitor:
    def __init__(self, avg_frames=30):
        self.prev_time = time.time()
        self.curr_fps = 0
        self.avg_frames = avg_frames
        self.fps_history = deque(maxlen=avg_frames)
        self.latency = 0
    
    def update(self, capture_time=None):
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        self.fps_history.append(fps)
        self.curr_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        # Calculate latency if capture time is provided
        if capture_time:
            self.latency = curr_time - capture_time
        
        return self.curr_fps
    
    def draw(self, frame):
        fps_text = f"FPS: {self.curr_fps:.1f}"
        latency_text = f"Latency: {self.latency*1000:.0f}ms"
        
        cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, latency_text, (frame.shape[1] - 200, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
                        <title>YOLOv8 Real-Time Stream</title>
                        <style>
                        body {
                            font-family: Arial, sans-serif;
                            margin: 0;
                            padding: 20px;
                            text-align: center;
                            background-color: #f0f0f0;
                        }
                        h1 {
                            color: #333;
                        }
                        img {
                            max-width: 100%;
                            border: 2px solid #333;
                            box-shadow: 0 0 10px rgba(0,0,0,0.2);
                        }
                        </style>
                        </head>
                        <body>
                        <h1>YOLOv8 Real-Time Detection Stream</h1>
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
                                else:
                                    time.sleep(0.01)  # No frame available yet
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
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
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

class FrameCapture:
    """Thread-based frame capture to minimize latency"""
    def __init__(self, rtmp_url, buffer_size=10, max_latency=0.5):
        self.rtmp_url = rtmp_url
        self.buffer_size = buffer_size
        self.max_latency = max_latency
        self.frame_queue = Queue(maxsize=buffer_size)
        self.running = False
        self.cap = None
        self.thread = None
        self.frame_count = 0
        self.dropped_frames = 0
    
    def start(self):
        """Start the capture thread"""
        self.running = True
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self.thread.is_alive()
    
    def _capture_loop(self):
        """Main capture loop that runs in a separate thread"""
        print(f"Opening RTMP stream: {self.rtmp_url}")
        self.cap = cv2.VideoCapture(self.rtmp_url)
        if not self.cap.isOpened():
            print(f"Error: Could not open RTMP stream: {self.rtmp_url}")
            self.running = False
            return
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Stream properties: {width}x{height}, {fps} FPS")
        
        reconnect_delay = 2.0
        connection_failures = 0
        max_failures = 10
        
        while self.running:
            try:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    connection_failures += 1
                    print(f"Lost connection to stream. Attempt {connection_failures}/{max_failures} to reconnect...")
                    
                    if connection_failures >= max_failures:
                        print("Maximum reconnection attempts reached. Exiting.")
                        self.running = False
                        break
                    
                    # Release and reopen capture
                    self.cap.release()
                    time.sleep(reconnect_delay)
                    self.cap = cv2.VideoCapture(self.rtmp_url)
                    if not self.cap.isOpened():
                        print(f"Failed to reconnect to stream: {self.rtmp_url}")
                        continue
                    else:
                        print("Successfully reconnected to stream")
                        connection_failures = 0
                else:
                    # Reset connection failures counter on successful frame
                    connection_failures = 0
                    self.frame_count += 1
                    
                    # Create a frame object with timestamp
                    frame_obj = {
                        'frame': frame,
                        'timestamp': time.time(),
                        'index': self.frame_count
                    }
                    
                    # Try to add to queue, drop old frames if full
                    try:
                        self.frame_queue.put_nowait(frame_obj)
                    except Full:
                        # Drop oldest frame if queue is full
                        try:
                            _ = self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame_obj)
                            self.dropped_frames += 1
                        except (Empty, Full):
                            # If we can't manage the queue properly, just continue
                            pass
            
            except Exception as e:
                print(f"Error in capture loop: {e}")
                time.sleep(1.0)
    
    def get_frame(self):
        """Get the next frame from the queue"""
        if not self.running or self.frame_queue.empty():
            return None
        
        # Get oldest frame in queue
        frame_obj = self.frame_queue.get()
        
        # If the frame is too old, keep getting newer frames
        current_time = time.time()
        while (current_time - frame_obj['timestamp']) > self.max_latency and not self.frame_queue.empty():
            self.dropped_frames += 1
            try:
                frame_obj = self.frame_queue.get_nowait()
            except Empty:
                break
        
        return frame_obj
    
    def release(self):
        """Stop the capture thread and release resources"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        if self.cap:
            self.cap.release()
        print(f"Capture ended. Processed {self.frame_count} frames. Dropped {self.dropped_frames} frames.")

def process_rtmp_stream(args):
    """Process RTMP stream with minimal latency"""
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
    counter = ObjectCounter(class_names) if args.count_objects else None
    
    # Initialize video streamer
    streamer = VideoStreamer(args.stream_port) if args.stream else None
    if streamer:
        streamer.start_server()
    
    # Initialize FPS monitor
    fps_monitor = FPSMonitor()
    
    # Initialize frame capturer
    capturer = FrameCapture(args.rtmp, args.buffer_size, args.max_latency)
    if not capturer.start():
        print("Failed to start frame capture")
        return
    
    # Process the stream
    frame_count = 0
    skip_count = 0
    processing_start = time.time()
    
    try:
        while True:
            # Get frame from capturer
            frame_obj = capturer.get_frame()
            if frame_obj is None:
                # No frame available yet
                time.sleep(0.01)
                continue
            
            frame = frame_obj['frame']
            capture_time = frame_obj['timestamp']
            frame_index = frame_obj['index']
            
            # Resize frame if requested
            if args.resize:
                orig_height, orig_width = frame.shape[:2]
                new_width = args.resize
                new_height = int(orig_height * (new_width / orig_width))
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Frame skipping logic for performance
            if args.skip_frames > 0:
                if skip_count < args.skip_frames:
                    skip_count += 1
                    
                    # Show previous annotated frame if available
                    if 'annotated_frame' in locals():
                        if args.show:
                            cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        if streamer:
                            streamer.update_frame(annotated_frame)
                    continue
                else:
                    skip_count = 0
            
            frame_count += 1
            
            # Run detection/tracking on the frame
            if args.track:
                results = model.track(frame, conf=args.conf, persist=True, verbose=False, half=args.half)
            else:
                results = model(frame, conf=args.conf, verbose=False, half=args.half)
            
            # Update object counter if enabled
            if counter:
                counter.update(results, frame_count)
            
            # Get the annotated frame
            annotated_frame = results[0].plot()
            
            # Add object count information if enabled
            if counter:
                annotated_frame = counter.draw_counts(annotated_frame)
            
            # Update and draw FPS/latency
            current_fps = fps_monitor.update(capture_time)
            annotated_frame = fps_monitor.draw(annotated_frame)
            
            # Add frame info
            cv2.putText(annotated_frame, f"Frame: {frame_index}", 
                        (10, annotated_frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update streaming frame if enabled
            if streamer:
                streamer.update_frame(annotated_frame)
            
            # Show the frame if requested
            if args.show:
                cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)
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
        fps_processing = frame_count / processing_time if processing_time > 0 else 0
        
        print(f"Processed {frame_count} frames in {processing_time:.2f} seconds")
        print(f"Processing speed: {fps_processing:.2f} FPS")
        
        # Print object counts if enabled
        if counter:
            print("\nTotal object counts:")
            for cls_name, count in sorted(counter.counts.items()):
                print(f"  {cls_name}: {count}")
        
        # Release resources
        capturer.release()
        if args.show:
            cv2.destroyAllWindows()
        
        if args.stream:
            print(f"Stream available at http://localhost:{args.stream_port}")

if __name__ == "__main__":
    args = setup_argparse()
    process_rtmp_stream(args) 