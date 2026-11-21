"""
Police Surveillance System Integration Module

This module integrates all surveillance components:
1. Automated License Plate Recognition (ALPR)
2. Vehicle Classification and Identification
3. Multi-Camera Vehicle Tracking
4. Vehicle Speed Estimation
"""
import cv2
import numpy as np
import torch
import os
import sys
import time
import json
import threading
import queue
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Local imports
from .. import config
from .alpr import ALPR
from .vehicle_classification import VehicleClassificationSystem
from .vehicle_tracking import MultiCameraTracker
from .speed_estimation import SpeedEstimator, AutoCalibrator


class SurveillanceSystem:
    """
    Integrated police surveillance system that combines all surveillance features
    """
    def __init__(self,
                 config_path: str = None, 
                 detection_model_path: str = None,
                 classifier_model_path: str = None,
                 identifier_model_path: str = None,
                 device: str = None):
        """
        Initialize the surveillance system
        
        Args:
            config_path: Path to configuration file
            detection_model_path: Path to object detection model
            classifier_model_path: Path to vehicle classification model
            identifier_model_path: Path to vehicle identification model
            device: Device to run inference on ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Surveillance system using device: {self.device}")
        
        # Load configuration if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            print("Using default configuration")
            
        # Initialize components with default values if not in config
        self.init_components(
            detection_model_path=detection_model_path,
            classifier_model_path=classifier_model_path,
            identifier_model_path=identifier_model_path
        )
        
        # Initialize frame tracking
        self.frame_count = 0
        self.processed_frames = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Initialize error and warning tracking
        self.errors = []
        self.warnings = []
        
        # Processing flags
        self.processing_enabled = True
        self.recording_enabled = False
        self.output_video = None
        
        # Create output directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        
        # Feature flags (enable/disable specific features)
        self.enable_alpr = self.config.get('enable_alpr', True)
        self.enable_classification = self.config.get('enable_classification', True)
        self.enable_tracking = self.config.get('enable_tracking', True)
        self.enable_speed_estimation = self.config.get('enable_speed_estimation', True)
        
        print("Surveillance system initialized with features:")
        print(f"- ALPR: {'Enabled' if self.enable_alpr else 'Disabled'}")
        print(f"- Classification: {'Enabled' if self.enable_classification else 'Disabled'}")
        print(f"- Tracking: {'Enabled' if self.enable_tracking else 'Disabled'}")
        print(f"- Speed estimation: {'Enabled' if self.enable_speed_estimation else 'Disabled'}")
        
    def init_components(self,
                       detection_model_path: str = None,
                       classifier_model_path: str = None,
                       identifier_model_path: str = None):
        """
        Initialize surveillance components
        
        Args:
            detection_model_path: Path to detection model
            classifier_model_path: Path to classification model
            identifier_model_path: Path to identification model
        """
        print("Initializing surveillance components...")
        
        # Initialize ALPR
        self.alpr = ALPR(
            detector_model_path=detection_model_path,
            confidence_threshold=self.config.get('alpr_confidence', 0.5),
            device=self.device
        )
        
        # Initialize Vehicle Classification System
        self.classification_system = VehicleClassificationSystem(
            classifier_model_path=classifier_model_path,
            identifier_model_path=identifier_model_path,
            confidence_threshold=self.config.get('classification_confidence', 0.5),
            similarity_threshold=self.config.get('identification_similarity', 0.7),
            device=self.device
        )
        
        # Initialize Multi-Camera Tracker
        self.tracker = MultiCameraTracker(
            identifier=self.classification_system.identifier,
            max_track_points=self.config.get('max_track_points', 100),
            track_timeout=self.config.get('track_timeout', 60.0),
            max_tracks=self.config.get('max_tracks', 1000)
        )
        
        # Initialize Speed Estimator
        speed_limit = self.config.get('speed_limit_mph', 70.0)
        self.speed_estimator = SpeedEstimator(
            pixels_per_meter=self.config.get('pixels_per_meter', 30.0),
            speed_limit_mph=speed_limit,
            min_track_points=self.config.get('min_track_points', 5),
            max_speed_mph=self.config.get('max_speed_mph', 150.0),
            smoothing_window=self.config.get('speed_smoothing_window', 3)
        )
        
        # Auto Calibrator for speed estimation
        self.auto_calibrator = AutoCalibrator(
            reference_vehicle_length_meters=self.config.get('reference_vehicle_length', 4.5),
            min_detections=self.config.get('min_calibration_detections', 20)
        )
        
        # Load calibration if available
        calibration_path = self.config.get('calibration_path', 'speed_calibration.json')
        if os.path.exists(calibration_path):
            self.speed_estimator.load_calibration(calibration_path)
        
        print("Surveillance components initialized")
    
    def process_frame(self, 
                     frame: np.ndarray, 
                     camera_id: str = "default",
                     timestamp: float = None,
                     draw_outputs: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame with all enabled surveillance features
        
        Args:
            frame: Input frame to process
            camera_id: Camera identifier
            timestamp: Frame timestamp (if None, current time is used)
            draw_outputs: Whether to draw results on the frame
            
        Returns:
            Tuple of (processed_frame, results_dict)
        """
        if timestamp is None:
            timestamp = time.time()
            
        if not self.processing_enabled:
            return frame, {}
            
        self.frame_count += 1
        
        # Initialize results dictionary
        results = {
            'frame_id': self.frame_count,
            'camera_id': camera_id,
            'timestamp': timestamp,
            'detections': [],
            'alpr_results': [],
            'classification_results': [],
            'speed_estimates': [],
            'processing_time': 0
        }
        
        # Measure processing time
        start_time = time.time()
        
        # Initialize the frame to draw on
        output_frame = frame.copy()
        
        try:
            # Step 1: License Plate Recognition (if enabled)
            if self.enable_alpr:
                alpr_results = self.alpr.process_frame(frame)
                results['alpr_results'] = alpr_results
                
                # Add to detections
                for result in alpr_results:
                    results['detections'].append({
                        'vehicle_bbox': result['vehicle_bbox'],
                        'vehicle_type': result['vehicle_type'],
                        'confidence': result['vehicle_confidence'],
                        'plate_text': result['plate_text'],
                        'plate_confidence': result['text_confidence']
                    })
                    
                # Draw ALPR results if requested
                if draw_outputs:
                    output_frame = self.alpr.draw_results(output_frame, alpr_results)
            
            # Step 2: Vehicle Classification (if enabled)
            if self.enable_classification:
                detections = results['detections']
                classification_results = self.classification_system.process_detections(frame, detections)
                results['classification_results'] = classification_results
                
                # Update detections with classification results
                if classification_results:
                    results['detections'] = classification_results
                    
                # Draw classification results if requested
                if draw_outputs:
                    output_frame = self.classification_system.draw_results(output_frame, classification_results)
            
            # Step 3: Vehicle Tracking (if enabled)
            if self.enable_tracking:
                detections = results['detections']
                tracked_detections = self.tracker.update(camera_id, detections, frame, timestamp)
                
                # Update detections with tracking info
                if tracked_detections:
                    results['detections'] = tracked_detections
                    
                # Draw tracking results if requested
                if draw_outputs:
                    output_frame = self.tracker.draw_tracks(output_frame, camera_id)
            
            # Step 4: Speed Estimation (if enabled)
            if self.enable_speed_estimation and self.enable_tracking:
                # Get active tracks for this camera
                active_tracks = self.tracker.get_active_tracks()
                
                # Estimate speeds
                speed_estimates = self.speed_estimator.process_tracks(active_tracks, camera_id, timestamp)
                results['speed_estimates'] = speed_estimates
                
                # Draw speed overlay if requested
                if draw_outputs and speed_estimates:
                    output_frame = self.speed_estimator.draw_speed_overlay(
                        output_frame, results['detections'], speed_estimates
                    )
            
            # Update FPS counter
            self.processed_frames += 1
            current_time = time.time()
            time_diff = current_time - self.last_fps_time
            
            if time_diff >= 1.0:  # Update FPS every second
                self.fps = self.processed_frames / time_diff
                self.processed_frames = 0
                self.last_fps_time = current_time
                
            # Draw FPS counter
            if draw_outputs:
                cv2.putText(
                    output_frame,
                    f"FPS: {self.fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
            # Record to video if enabled
            if self.recording_enabled and self.output_video is not None:
                self.output_video.write(output_frame)
                
        except Exception as e:
            # Log error and return original frame
            error_msg = f"Error processing frame: {str(e)}"
            print(error_msg)
            self.errors.append({
                'timestamp': timestamp,
                'message': error_msg,
                'frame_id': self.frame_count,
                'camera_id': camera_id
            })
            
            # Draw error message
            if draw_outputs:
                cv2.putText(
                    output_frame,
                    "ERROR: See logs",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        return output_frame, results

    def start_recording(self, 
                       output_path: str = None, 
                       fps: float = 30.0, 
                       resolution: Tuple[int, int] = None):
        """
        Start recording processed frames to video
        
        Args:
            output_path: Path to save the output video
            fps: Frames per second
            resolution: Output resolution (width, height)
        """
        if output_path is None:
            output_path = f"outputs/surveillance_{int(time.time())}.mp4"
            
        if resolution is None:
            resolution = (1280, 720)
            
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        self.recording_enabled = True
        
        print(f"Started recording to {output_path}")
        
    def stop_recording(self):
        """
        Stop recording
        """
        if self.recording_enabled and self.output_video is not None:
            self.output_video.release()
            self.output_video = None
            self.recording_enabled = False
            print("Recording stopped")
            
    def save_config(self, config_path: str = "surveillance_config.json"):
        """
        Save configuration to file
        
        Args:
            config_path: Path to save configuration
        """
        config = {
            # System settings
            'enable_alpr': self.enable_alpr,
            'enable_classification': self.enable_classification,
            'enable_tracking': self.enable_tracking,
            'enable_speed_estimation': self.enable_speed_estimation,
            
            # ALPR settings
            'alpr_confidence': self.alpr.detector.confidence_threshold,
            
            # Classification settings
            'classification_confidence': self.classification_system.classifier.confidence_threshold,
            'identification_similarity': self.classification_system.identifier.similarity_threshold,
            
            # Tracking settings
            'max_track_points': self.tracker.max_track_points,
            'track_timeout': self.tracker.track_timeout,
            'max_tracks': self.tracker.max_tracks,
            
            # Speed estimation settings
            'pixels_per_meter': self.speed_estimator.pixels_per_meter,
            'speed_limit_mph': self.speed_estimator.speed_limit_mph,
            'min_track_points': self.speed_estimator.min_track_points,
            'max_speed_mph': self.speed_estimator.max_speed_mph,
            'speed_smoothing_window': self.speed_estimator.smoothing_window,
            'calibration_path': 'speed_calibration.json',
            
            # Auto calibration settings
            'reference_vehicle_length': self.auto_calibrator.reference_length,
            'min_calibration_detections': self.auto_calibrator.min_detections,
            
            # Timestamp for this configuration
            'timestamp': time.time()
        }
        
        # Save to file
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"Configuration saved to {config_path}")
        
    def load_config(self, config_path: str):
        """
        Load configuration from file
        
        Args:
            config_path: Path to load configuration
        """
        if not os.path.exists(config_path):
            print(f"Warning: Configuration file does not exist: {config_path}")
            return
            
        # Load from file
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Update feature flags
        self.enable_alpr = self.config.get('enable_alpr', self.enable_alpr)
        self.enable_classification = self.config.get('enable_classification', self.enable_classification)
        self.enable_tracking = self.config.get('enable_tracking', self.enable_tracking)
        self.enable_speed_estimation = self.config.get('enable_speed_estimation', self.enable_speed_estimation)
        
        # Reinitialize components with loaded config
        self.init_components()
        
        # Load speed calibration if specified
        calibration_path = self.config.get('calibration_path')
        if calibration_path and os.path.exists(calibration_path):
            self.speed_estimator.load_calibration(calibration_path)
            
        print(f"Configuration loaded from {config_path}")
        
    def toggle_feature(self, feature_name: str, enabled: bool = None):
        """
        Toggle a surveillance feature on or off
        
        Args:
            feature_name: Name of the feature ('alpr', 'classification', 'tracking', 'speed')
            enabled: True to enable, False to disable, None to toggle
        """
        feature_map = {
            'alpr': 'enable_alpr',
            'classification': 'enable_classification', 
            'tracking': 'enable_tracking',
            'speed': 'enable_speed_estimation'
        }
        
        if feature_name.lower() not in feature_map:
            print(f"Unknown feature: {feature_name}")
            return
            
        attr_name = feature_map[feature_name.lower()]
        
        if enabled is None:
            # Toggle the feature
            current = getattr(self, attr_name)
            setattr(self, attr_name, not current)
            print(f"Feature '{feature_name}' {'enabled' if not current else 'disabled'}")
        else:
            # Set to specified value
            setattr(self, attr_name, enabled)
            print(f"Feature '{feature_name}' {'enabled' if enabled else 'disabled'}")


class RTMPSurveillanceServer:
    """
    RTMP server for surveillance video streams
    """
    def __init__(self, 
                 surveillance_system: SurveillanceSystem,
                 rtmp_url: str = "rtmp://localhost/live/stream",
                 input_resolution: Tuple[int, int] = (1280, 720),
                 output_resolution: Tuple[int, int] = (1280, 720),
                 fps: float = 30.0,
                 buffer_size: int = 10):
        """
        Initialize RTMP surveillance server
        
        Args:
            surveillance_system: SurveillanceSystem instance
            rtmp_url: RTMP server URL
            input_resolution: Input video resolution
            output_resolution: Output video resolution
            fps: Frames per second
            buffer_size: Frame buffer size
        """
        self.surveillance = surveillance_system
        self.rtmp_url = rtmp_url
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.fps = fps
        
        # Initialize frame buffer
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        
        # Initialize output stream
        self.output_stream = None
        self.streaming = False
        
        # Processing thread
        self.processing_thread = None
        self.stop_event = threading.Event()
        
    def start_streaming(self):
        """
        Start the RTMP stream
        """
        if self.streaming:
            print("Streaming already started")
            return
            
        # Initialize output stream
        width, height = self.output_resolution
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{width}x{height}",
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-f', 'flv',
            self.rtmp_url
        ]
        
        # Start ffmpeg process
        import subprocess
        self.output_stream = subprocess.Popen(
            command, stdin=subprocess.PIPE
        )
        
        # Start processing thread
        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.streaming = True
        print(f"Started streaming to {self.rtmp_url}")
        
    def stop_streaming(self):
        """
        Stop the RTMP stream
        """
        if not self.streaming:
            return
            
        # Stop processing thread
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
            
        # Close output stream
        if self.output_stream:
            self.output_stream.stdin.close()
            self.output_stream.wait()
            self.output_stream = None
            
        self.streaming = False
        print("Streaming stopped")
        
    def add_frame(self, frame: np.ndarray, camera_id: str = "default"):
        """
        Add a frame to the processing queue
        
        Args:
            frame: Input frame
            camera_id: Camera identifier
        """
        # Resize frame to input resolution if needed
        if frame.shape[1] != self.input_resolution[0] or frame.shape[0] != self.input_resolution[1]:
            frame = cv2.resize(frame, self.input_resolution)
            
        # Add to buffer, drop frames if buffer is full
        try:
            self.frame_buffer.put_nowait((frame, camera_id, time.time()))
        except queue.Full:
            # Drop frame if buffer is full
            pass
            
    def _processing_loop(self):
        """
        Main processing loop
        """
        while not self.stop_event.is_set():
            try:
                # Get frame from buffer
                frame, camera_id, timestamp = self.frame_buffer.get(timeout=1)
                
                # Process frame
                processed_frame, results = self.surveillance.process_frame(
                    frame, camera_id, timestamp, draw_outputs=True
                )
                
                # Resize to output resolution if needed
                if processed_frame.shape[1] != self.output_resolution[0] or processed_frame.shape[0] != self.output_resolution[1]:
                    processed_frame = cv2.resize(processed_frame, self.output_resolution)
                    
                # Write to output stream
                if self.output_stream and self.output_stream.poll() is None:
                    self.output_stream.stdin.write(processed_frame.tobytes())
                    
            except queue.Empty:
                # No frames in buffer
                continue
            except Exception as e:
                print(f"Error in processing loop: {str(e)}")
                
    def is_streaming(self) -> bool:
        """
        Check if streaming is active
        
        Returns:
            True if streaming is active
        """
        return self.streaming


# Example usage
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Police Surveillance System")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live feed")
    parser.add_argument("--config", type=str, default="surveillance_config.json", help="Path to configuration file")
    parser.add_argument("--output", type=str, help="Path to output video file")
    parser.add_argument("--rtmp", action="store_true", help="Enable RTMP streaming")
    parser.add_argument("--rtmp-url", type=str, default="rtmp://localhost/live/stream", help="RTMP URL")
    
    # Feature flags
    parser.add_argument("--disable-alpr", action="store_true", help="Disable ALPR")
    parser.add_argument("--disable-classification", action="store_true", help="Disable classification")
    parser.add_argument("--disable-tracking", action="store_true", help="Disable tracking")
    parser.add_argument("--disable-speed", action="store_true", help="Disable speed estimation")
    
    args = parser.parse_args()
    
    # Initialize surveillance system
    system = SurveillanceSystem(config_path=args.config if os.path.exists(args.config) else None)
    
    # Apply feature flags
    if args.disable_alpr:
        system.toggle_feature('alpr', False)
    if args.disable_classification:
        system.toggle_feature('classification', False)
    if args.disable_tracking:
        system.toggle_feature('tracking', False)
    if args.disable_speed:
        system.toggle_feature('speed', False)
    
    # Initialize RTMP server if requested
    rtmp_server = None
    if args.rtmp:
        rtmp_server = RTMPSurveillanceServer(
            surveillance_system=system,
            rtmp_url=args.rtmp_url
        )
        rtmp_server.start_streaming()
    
    # Start recording if output specified
    if args.output:
        system.start_recording(args.output)
    
    # Open video source
    if args.video and os.path.exists(args.video):
        cap = cv2.VideoCapture(args.video)
        source_name = os.path.basename(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"Camera {args.camera}"
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source_name}")
        sys.exit(1)
    
    # Process video frames
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame, results = system.process_frame(frame, camera_id=source_name)
            
            # Add to RTMP server if enabled
            if rtmp_server and rtmp_server.is_streaming():
                rtmp_server.add_frame(processed_frame, source_name)
                
            # Display result
            cv2.imshow("Police Surveillance System", processed_frame)
            
            # Check for exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            # Toggle features with keyboard shortcuts
            if key == ord('a'):
                system.toggle_feature('alpr')
            elif key == ord('c'):
                system.toggle_feature('classification')
            elif key == ord('t'):
                system.toggle_feature('tracking')
            elif key == ord('s'):
                system.toggle_feature('speed')
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        if system.recording_enabled:
            system.stop_recording()
            
        if rtmp_server and rtmp_server.is_streaming():
            rtmp_server.stop_streaming()
            
        # Save configuration
        system.save_config(args.config)
        
        print("System shutdown complete") 