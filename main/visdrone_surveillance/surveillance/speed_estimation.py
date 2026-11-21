"""
Vehicle Speed Estimation Module
"""
import cv2
import numpy as np
import math
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, deque

# Local imports
from .. import config
from .vehicle_tracking import TrackingPoint, VehicleTrack


class SpeedEstimator:
    """
    Estimates vehicle speeds from video feeds
    """
    def __init__(self,
                 calibration_data: Dict = None,
                 pixels_per_meter: float = None,
                 speed_limit_mph: float = 70.0,
                 min_track_points: int = 5,
                 max_speed_mph: float = 150.0,
                 smoothing_window: int = 3):
        """
        Initialize speed estimator
        
        Args:
            calibration_data: Camera calibration data
            pixels_per_meter: Conversion factor from pixels to meters (if known)
            speed_limit_mph: Speed limit for highlighting violations
            min_track_points: Minimum number of tracking points required for estimation
            max_speed_mph: Maximum plausible speed (for filtering errors)
            smoothing_window: Window size for speed smoothing
        """
        self.calibration_data = calibration_data or {}
        self.pixels_per_meter = pixels_per_meter
        self.speed_limit_mph = speed_limit_mph
        self.min_track_points = min_track_points
        self.max_speed_mph = max_speed_mph
        self.smoothing_window = smoothing_window
        
        # Store camera calibrations
        self.camera_calibrations = {}  # camera_id -> calibration data
        
        # Store speed estimates
        self.vehicle_speeds = defaultdict(list)  # vehicle_id -> list of (timestamp, speed_mph)
        
        # Initialize calibration zones if data provided
        if calibration_data:
            self._init_calibrations_from_data(calibration_data)
    
    def _init_calibrations_from_data(self, calibration_data: Dict) -> None:
        """
        Initialize calibrations from provided data
        
        Args:
            calibration_data: Dict with camera calibration data
        """
        for camera_id, data in calibration_data.items():
            self.camera_calibrations[camera_id] = data
            
    def calibrate_camera(self, 
                        camera_id: str, 
                        reference_points: List[Tuple[Tuple[int, int], float]],
                        reference_distance_meters: float = None) -> None:
        """
        Calibrate camera with known reference distances
        
        Args:
            camera_id: Camera identifier
            reference_points: List of (pixel_position, real_world_position)
            reference_distance_meters: Known real-world distance
        """
        # Calculate pixels per meter
        if reference_distance_meters and len(reference_points) >= 2:
            p1, _ = reference_points[0]
            p2, _ = reference_points[1]
            pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            pixels_per_meter = pixel_distance / reference_distance_meters
            
            calibration = {
                'pixels_per_meter': pixels_per_meter,
                'reference_points': reference_points,
                'reference_distance_meters': reference_distance_meters,
                'calibration_date': time.time()
            }
            
            self.camera_calibrations[camera_id] = calibration
            print(f"Camera {camera_id} calibrated: {pixels_per_meter:.2f} pixels/meter")
        else:
            # For demo, use a default value
            pixels_per_meter = self.pixels_per_meter or 30.0  # Approximate value
            
            calibration = {
                'pixels_per_meter': pixels_per_meter,
                'calibration_date': time.time(),
                'is_default': True
            }
            
            self.camera_calibrations[camera_id] = calibration
            print(f"Camera {camera_id} using default calibration: {pixels_per_meter:.2f} pixels/meter")
    
    def _get_pixels_per_meter(self, camera_id: str) -> float:
        """
        Get pixels per meter for a specific camera
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Pixels per meter conversion factor
        """
        # If camera is calibrated, use its value
        if camera_id in self.camera_calibrations:
            return self.camera_calibrations[camera_id].get('pixels_per_meter', 30.0)
            
        # Otherwise use default value
        return self.pixels_per_meter or 30.0
        
    def estimate_speed(self, 
                      track: VehicleTrack,
                      camera_id: str,
                      current_timestamp: float = None) -> Dict:
        """
        Estimate vehicle speed from tracking data
        
        Args:
            track: Vehicle tracking data
            camera_id: Camera identifier
            current_timestamp: Current timestamp (if None, use latest point timestamp)
            
        Returns:
            Dictionary with speed estimates
        """
        # Get points for this camera
        camera_points = track.get_points_by_camera(camera_id)
        
        # Need minimum number of points
        if len(camera_points) < self.min_track_points:
            return {
                'vehicle_id': track.vehicle_id,
                'speed_mph': 0.0,
                'confidence': 0.0,
                'sufficient_data': False
            }
            
        # Get pixels per meter for this camera
        pixels_per_meter = self._get_pixels_per_meter(camera_id)
        
        # Calculate time intervals and distances
        speeds = []
        
        for i in range(1, len(camera_points)):
            p1 = camera_points[i-1]
            p2 = camera_points[i]
            
            # Time difference in seconds
            dt = p2.timestamp - p1.timestamp
            
            # Need non-zero time difference
            if dt <= 0:
                continue
                
            # Distance in pixels
            distance_px = p1.distance_to(p2)
            
            # Convert to meters
            distance_meters = distance_px / pixels_per_meter
            
            # Speed in meters per second
            speed_mps = distance_meters / dt
            
            # Convert to mph
            speed_mph = speed_mps * 2.23694  # m/s to mph
            
            # Apply basic sanity check
            if 0 < speed_mph < self.max_speed_mph:
                speeds.append((p2.timestamp, speed_mph))
        
        # No valid speed measurements
        if not speeds:
            return {
                'vehicle_id': track.vehicle_id,
                'speed_mph': 0.0,
                'confidence': 0.0,
                'sufficient_data': False
            }
            
        # Store speed measurements
        vehicle_id = track.vehicle_id
        self.vehicle_speeds[vehicle_id].extend(speeds)
        
        # Keep only recent measurements
        recent_cutoff = time.time() - 60  # Last minute
        self.vehicle_speeds[vehicle_id] = [
            s for s in self.vehicle_speeds[vehicle_id] if s[0] >= recent_cutoff
        ]
        
        # Calculate current speed using smoothing window
        if self.smoothing_window > 0 and len(speeds) >= self.smoothing_window:
            # Use last N measurements
            recent_speeds = [s[1] for s in speeds[-self.smoothing_window:]]
            current_speed = sum(recent_speeds) / len(recent_speeds)
        else:
            # Use latest measurement
            current_speed = speeds[-1][1]
            
        # Calculate speed statistics
        if len(speeds) >= 3:
            all_speeds = [s[1] for s in speeds]
            avg_speed = sum(all_speeds) / len(all_speeds)
            min_speed = min(all_speeds)
            max_speed = max(all_speeds)
            std_dev = np.std(all_speeds)
            
            # Higher confidence with more measurements and lower std dev
            confidence = min(0.95, 0.5 + 0.5 * (1 - min(1.0, std_dev / avg_speed if avg_speed > 0 else 1.0)))
        else:
            avg_speed = current_speed
            min_speed = current_speed
            max_speed = current_speed
            std_dev = 0.0
            confidence = 0.5  # Medium confidence
            
        # Check if exceeding speed limit
        is_speeding = current_speed > self.speed_limit_mph
        
        return {
            'vehicle_id': vehicle_id,
            'speed_mph': round(current_speed, 1),
            'avg_speed_mph': round(avg_speed, 1),
            'min_speed_mph': round(min_speed, 1),
            'max_speed_mph': round(max_speed, 1),
            'std_dev': round(std_dev, 2),
            'confidence': round(confidence, 2),
            'is_speeding': is_speeding,
            'speed_limit_mph': self.speed_limit_mph,
            'sufficient_data': True,
            'measurements': len(speeds)
        }
    
    def process_tracks(self, 
                      tracks: List[VehicleTrack],
                      camera_id: str,
                      current_timestamp: float = None) -> List[Dict]:
        """
        Process multiple tracks to get speed estimates
        
        Args:
            tracks: List of vehicle tracks
            camera_id: Camera identifier
            current_timestamp: Current timestamp
            
        Returns:
            List of speed estimation results
        """
        results = []
        
        for track in tracks:
            # Skip tracks with insufficient data
            camera_points = track.get_points_by_camera(camera_id)
            if len(camera_points) < self.min_track_points:
                continue
                
            # Estimate speed
            speed_data = self.estimate_speed(track, camera_id, current_timestamp)
            
            # Only include if we have sufficient data
            if speed_data['sufficient_data']:
                results.append(speed_data)
                
        return results
    
    def draw_speed_overlay(self, 
                          frame: np.ndarray,
                          vehicle_detections: List[Dict],
                          speed_estimates: List[Dict]) -> np.ndarray:
        """
        Draw speed information overlay on frame
        
        Args:
            frame: Input frame
            vehicle_detections: List of vehicle detections
            speed_estimates: List of speed estimation results
            
        Returns:
            Frame with speed overlay
        """
        img = frame.copy()
        
        # Create mapping from vehicle_id to speed data
        speed_map = {s['vehicle_id']: s for s in speed_estimates}
        
        # Process each detection
        for detection in vehicle_detections:
            vehicle_id = detection.get('vehicle_id')
            
            # Skip if no vehicle ID or speed data
            if not vehicle_id or vehicle_id not in speed_map:
                continue
                
            # Get speed data
            speed_data = speed_map[vehicle_id]
            
            # Get bounding box
            bbox = detection.get('vehicle_bbox') or detection.get('bbox')
            if not bbox:
                continue
                
            x1, y1, x2, y2 = bbox
            
            # Choose color based on speeding
            is_speeding = speed_data.get('is_speeding', False)
            color = (0, 0, 255) if is_speeding else (0, 255, 0)
            
            # Draw speed
            speed_mph = speed_data.get('speed_mph', 0)
            confidence = speed_data.get('confidence', 0)
            
            cv2.putText(
                img,
                f"{speed_mph} mph ({confidence:.2f})",
                (x1, y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            # Draw warning for speeding
            if is_speeding:
                cv2.putText(
                    img,
                    "EXCEEDING SPEED LIMIT",
                    (x1, y2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
                
        return img
    
    def save_calibration(self, filepath: str) -> None:
        """
        Save camera calibration data to file
        
        Args:
            filepath: Path to save file
        """
        # Convert calibration data to JSON-serializable format
        serializable_calibrations = {}
        
        for camera_id, calibration in self.camera_calibrations.items():
            # Handle reference points that might contain tuples
            cal_copy = calibration.copy()
            
            if 'reference_points' in cal_copy:
                # Convert tuple points to lists
                cal_copy['reference_points'] = [
                    ([p[0][0], p[0][1]], p[1]) for p in cal_copy['reference_points']
                ]
                
            serializable_calibrations[camera_id] = cal_copy
            
        # Save to file
        with open(filepath, 'w') as f:
            json.dump({
                'camera_calibrations': serializable_calibrations,
                'default_pixels_per_meter': self.pixels_per_meter,
                'speed_limit_mph': self.speed_limit_mph,
                'timestamp': time.time()
            }, f, indent=2)
            
        print(f"Calibration data saved to {filepath}")
    
    def load_calibration(self, filepath: str) -> None:
        """
        Load camera calibration data from file
        
        Args:
            filepath: Path to load file
        """
        if not os.path.exists(filepath):
            print(f"Warning: Calibration file does not exist: {filepath}")
            return
            
        # Load from file
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Update calibration data
        self.camera_calibrations = {}
        
        for camera_id, calibration in data.get('camera_calibrations', {}).items():
            # Convert reference points back to tuples if present
            if 'reference_points' in calibration:
                calibration['reference_points'] = [
                    ((p[0][0], p[0][1]), p[1]) for p in calibration['reference_points']
                ]
                
            self.camera_calibrations[camera_id] = calibration
            
        # Update other parameters
        self.pixels_per_meter = data.get('default_pixels_per_meter', self.pixels_per_meter)
        self.speed_limit_mph = data.get('speed_limit_mph', self.speed_limit_mph)
        
        print(f"Loaded calibration data from {filepath}")


class AutoCalibrator:
    """
    Automatic camera calibration for speed estimation
    """
    def __init__(self,
                 reference_vehicle_length_meters: float = 4.5,
                 min_detections: int = 20,
                 max_detections: int = 100):
        """
        Initialize auto-calibrator
        
        Args:
            reference_vehicle_length_meters: Reference length for standard vehicle
            min_detections: Minimum detections needed for calibration
            max_detections: Maximum detections to use for calibration
        """
        self.reference_length = reference_vehicle_length_meters
        self.min_detections = min_detections
        self.max_detections = max_detections
        
        # Store vehicle detections
        self.detections = []
        
    def add_detection(self, 
                     bbox: Tuple[int, int, int, int],
                     vehicle_type: str) -> None:
        """
        Add a vehicle detection for calibration purposes
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            vehicle_type: Type of vehicle
        """
        # Only store if within limit
        if len(self.detections) < self.max_detections:
            vehicle_length_meters = self._get_reference_length(vehicle_type)
            
            x1, y1, x2, y2 = bbox
            width_px = x2 - x1
            height_px = y2 - y1
            
            # Assuming vehicle length corresponds to width in image
            self.detections.append({
                'bbox': bbox,
                'width_px': width_px,
                'height_px': height_px,
                'vehicle_type': vehicle_type,
                'reference_length_meters': vehicle_length_meters
            })
    
    def _get_reference_length(self, vehicle_type: str) -> float:
        """
        Get reference length for a vehicle type
        
        Args:
            vehicle_type: Type of vehicle
            
        Returns:
            Reference length in meters
        """
        # Average vehicle lengths
        lengths = {
            'sedan': 4.5,
            'suv': 4.7,
            'truck': 6.0,
            'van': 5.0,
            'pickup': 5.4,
            'bus': 12.0,
            'motorcycle': 2.2
        }
        
        return lengths.get(vehicle_type.lower(), self.reference_length)
    
    def calibrate(self) -> Optional[float]:
        """
        Perform automatic calibration
        
        Returns:
            Estimated pixels per meter or None if insufficient data
        """
        if len(self.detections) < self.min_detections:
            print(f"Insufficient detections for calibration: {len(self.detections)}/{self.min_detections}")
            return None
            
        # Calculate pixels per meter for each detection
        pixels_per_meter_values = []
        
        for detection in self.detections:
            width_px = detection['width_px']
            length_m = detection['reference_length_meters']
            
            # Calculate pixels per meter
            ppm = width_px / length_m
            pixels_per_meter_values.append(ppm)
            
        # Use median to reduce outlier impact
        pixels_per_meter = np.median(pixels_per_meter_values)
        
        print(f"Auto-calibration complete: {pixels_per_meter:.2f} pixels/meter " +
              f"(min: {min(pixels_per_meter_values):.2f}, " +
              f"max: {max(pixels_per_meter_values):.2f}, " +
              f"std: {np.std(pixels_per_meter_values):.2f})")
        
        return pixels_per_meter
    
    def reset(self) -> None:
        """
        Reset calibration data
        """
        self.detections = []


# Example usage
if __name__ == "__main__":
    import sys
    
    # Create speed estimator with default calibration
    estimator = SpeedEstimator(
        pixels_per_meter=30.0,  # Example value
        speed_limit_mph=55.0
    )
    
    # Example of manual calibration
    estimator.calibrate_camera(
        camera_id="cam1",
        reference_points=[((100, 100), 0), ((300, 100), 20)],  # 20 meters reference
        reference_distance_meters=20.0
    )
    
    # Create a simulated vehicle track
    track = VehicleTrack("vehicle1", max_points=100)
    
    # Add simulated tracking points (covering 10 meters at 36 mph = 16 m/s)
    base_time = time.time()
    pixels_per_meter = 30.0
    
    for i in range(10):
        # Position advances 1.6 meters (at 36 mph) every 0.1 seconds
        # 1.6 meters * 30 pixels/meter = 48 pixels
        x = 100 + i * 48
        y = 200
        
        # Create tracking point
        point = TrackingPoint(
            vehicle_id="vehicle1",
            frame_id=i,
            camera_id="cam1",
            bbox=(x - 30, y - 20, x + 30, y + 20),
            timestamp=base_time + i * 0.1,
            metadata={'vehicle_type': 'sedan'}
        )
        
        # Add to track
        track.add_point(point)
    
    # Estimate speed
    speed_data = estimator.estimate_speed(track, "cam1")
    
    # Print result
    print(f"Speed estimate: {speed_data['speed_mph']} mph " +
          f"(confidence: {speed_data['confidence']:.2f})")
    
    # Save calibration data
    estimator.save_calibration("speed_calibration.json") 