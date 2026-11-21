"""
Multi-Camera Vehicle Tracking Module
"""
import cv2
import numpy as np
import torch
import os
import time
import uuid
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set
from collections import defaultdict, deque

# Local imports
from .. import config
from .vehicle_classification import VehicleIdentifier


class TrackingPoint:
    """
    Represents a single tracking point for a vehicle
    """
    def __init__(self, 
                 vehicle_id: str,
                 frame_id: int,
                 camera_id: str,
                 bbox: Tuple[int, int, int, int],
                 timestamp: float,
                 metadata: Dict = None):
        """
        Initialize tracking point
        
        Args:
            vehicle_id: Unique ID of the vehicle
            frame_id: Frame number
            camera_id: Camera identifier
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            timestamp: Unix timestamp
            metadata: Additional metadata
        """
        self.vehicle_id = vehicle_id
        self.frame_id = frame_id
        self.camera_id = camera_id
        self.bbox = bbox
        self.timestamp = timestamp
        self.metadata = metadata or {}
        
        # Compute center point
        x1, y1, x2, y2 = bbox
        self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
    def distance_to(self, other: 'TrackingPoint') -> float:
        """
        Calculate Euclidean distance to another tracking point
        
        Args:
            other: Another tracking point
            
        Returns:
            Euclidean distance
        """
        if self.camera_id != other.camera_id:
            return float('inf')  # Cannot directly compare across cameras
            
        return np.sqrt(
            (self.center[0] - other.center[0]) ** 2 + 
            (self.center[1] - other.center[1]) ** 2
        )
    
    def to_dict(self) -> Dict:
        """
        Convert tracking point to dictionary
        
        Returns:
            Dictionary representation
        """
        return {
            'vehicle_id': self.vehicle_id,
            'frame_id': self.frame_id,
            'camera_id': self.camera_id,
            'bbox': self.bbox,
            'center': self.center,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrackingPoint':
        """
        Create tracking point from dictionary
        
        Args:
            data: Dictionary representation
            
        Returns:
            TrackingPoint object
        """
        return cls(
            vehicle_id=data['vehicle_id'],
            frame_id=data['frame_id'],
            camera_id=data['camera_id'],
            bbox=tuple(data['bbox']),
            timestamp=data['timestamp'],
            metadata=data.get('metadata', {})
        )


class VehicleTrack:
    """
    Represents a track for a specific vehicle
    """
    def __init__(self, vehicle_id: str, max_points: int = 100):
        """
        Initialize vehicle track
        
        Args:
            vehicle_id: Unique ID of the vehicle
            max_points: Maximum number of tracking points to store
        """
        self.vehicle_id = vehicle_id
        self.max_points = max_points
        self.points = deque(maxlen=max_points)
        self.last_seen = 0
        self.first_seen = float('inf')
        self.cameras_seen = set()
        self.metadata = {}
        
    def add_point(self, point: TrackingPoint) -> None:
        """
        Add a tracking point
        
        Args:
            point: TrackingPoint to add
        """
        self.points.append(point)
        self.last_seen = max(self.last_seen, point.timestamp)
        self.first_seen = min(self.first_seen, point.timestamp)
        self.cameras_seen.add(point.camera_id)
        
        # Update metadata with the most recent info
        if point.metadata:
            self.metadata.update(point.metadata)
    
    def get_last_point(self) -> Optional[TrackingPoint]:
        """
        Get the most recent tracking point
        
        Returns:
            Most recent TrackingPoint or None if track is empty
        """
        if not self.points:
            return None
        return self.points[-1]
    
    def get_points_by_camera(self, camera_id: str) -> List[TrackingPoint]:
        """
        Get all points from a specific camera
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            List of TrackingPoints from the specified camera
        """
        return [p for p in self.points if p.camera_id == camera_id]
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """
        Get the trajectory as a list of center points
        
        Returns:
            List of (x, y) center points
        """
        return [p.center for p in self.points]
    
    def predict_next_position(self, camera_id: str) -> Optional[Tuple[int, int]]:
        """
        Predict the next position based on recent movement
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Predicted (x, y) position or None if prediction isn't possible
        """
        # Get points from the specified camera
        camera_points = self.get_points_by_camera(camera_id)
        
        # Need at least 2 points to predict
        if len(camera_points) < 2:
            return None
            
        # Use last two points to calculate velocity
        p1 = camera_points[-2]
        p2 = camera_points[-1]
        
        # Calculate displacement vector
        dx = p2.center[0] - p1.center[0]
        dy = p2.center[1] - p1.center[1]
        
        # If no time difference, can't predict
        dt = p2.timestamp - p1.timestamp
        if dt == 0:
            return None
            
        # Calculate velocity
        vx = dx / dt
        vy = dy / dt
        
        # Predict position after 1 second
        pred_x = p2.center[0] + int(vx * 1)
        pred_y = p2.center[1] + int(vy * 1)
        
        return (pred_x, pred_y)
    
    def to_dict(self) -> Dict:
        """
        Convert track to dictionary
        
        Returns:
            Dictionary representation
        """
        return {
            'vehicle_id': self.vehicle_id,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'cameras_seen': list(self.cameras_seen),
            'metadata': self.metadata,
            'points': [p.to_dict() for p in self.points]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VehicleTrack':
        """
        Create track from dictionary
        
        Args:
            data: Dictionary representation
            
        Returns:
            VehicleTrack object
        """
        track = cls(vehicle_id=data['vehicle_id'])
        track.first_seen = data['first_seen']
        track.last_seen = data['last_seen']
        track.cameras_seen = set(data['cameras_seen'])
        track.metadata = data.get('metadata', {})
        
        # Create points from dictionary
        for point_data in data.get('points', []):
            track.add_point(TrackingPoint.from_dict(point_data))
            
        return track


class MultiCameraTracker:
    """
    Track vehicles across multiple cameras
    """
    def __init__(self,
                 identifier: VehicleIdentifier = None,
                 max_track_points: int = 100,
                 track_timeout: float = 60.0,
                 max_tracks: int = 1000):
        """
        Initialize multi-camera tracker
        
        Args:
            identifier: Vehicle identifier for re-identification
            max_track_points: Maximum tracking points per vehicle
            track_timeout: Time in seconds after which a track is considered inactive
            max_tracks: Maximum number of tracks to store
        """
        self.identifier = identifier
        self.max_track_points = max_track_points
        self.track_timeout = track_timeout
        self.max_tracks = max_tracks
        
        # Track storage
        self.active_tracks = {}  # vehicle_id -> VehicleTrack
        self.inactive_tracks = {}  # vehicle_id -> VehicleTrack
        
        # Camera mapping (used to model spatial relationships between cameras)
        self.camera_transitions = defaultdict(set)  # (camera_id_a, camera_id_b) -> count
        
        # Frame counters
        self.frame_counters = defaultdict(int)  # camera_id -> frame_count
        
    def update(self, 
              camera_id: str, 
              detections: List[Dict],
              frame: np.ndarray,
              timestamp: float = None) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            camera_id: Camera identifier
            detections: List of vehicle detections
            frame: Video frame
            timestamp: Unix timestamp (if None, current time is used)
            
        Returns:
            List of updated detections with tracking info
        """
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = time.time()
            
        # Increment frame counter for this camera
        self.frame_counters[camera_id] += 1
        frame_id = self.frame_counters[camera_id]
        
        # Process each detection
        updated_detections = []
        for detection in detections:
            # Extract vehicle ID and bbox
            vehicle_id = detection.get('vehicle_id')
            bbox = detection.get('vehicle_bbox') or detection.get('bbox')
            
            if not vehicle_id or not bbox:
                # If no vehicle ID, try to identify using the identifier
                if self.identifier and bbox:
                    vehicle_id, _ = self.identifier.identify(frame, bbox, detection)
                    detection['vehicle_id'] = vehicle_id
                else:
                    # Skip this detection if we can't identify the vehicle
                    continue
            
            # Create tracking point
            point = TrackingPoint(
                vehicle_id=vehicle_id,
                frame_id=frame_id,
                camera_id=camera_id,
                bbox=bbox,
                timestamp=timestamp,
                metadata=detection
            )
            
            # Update or create track
            if vehicle_id in self.active_tracks:
                # Update existing track
                track = self.active_tracks[vehicle_id]
                
                # Check if vehicle appeared in a new camera
                prev_point = track.get_last_point()
                if prev_point and prev_point.camera_id != camera_id:
                    # Register camera transition
                    self._register_camera_transition(prev_point.camera_id, camera_id)
                    detection['camera_transition'] = True
                    detection['previous_camera'] = prev_point.camera_id
                else:
                    detection['camera_transition'] = False
                
                # Add point to track
                track.add_point(point)
            elif vehicle_id in self.inactive_tracks:
                # Reactivate inactive track
                track = self.inactive_tracks.pop(vehicle_id)
                track.add_point(point)
                self.active_tracks[vehicle_id] = track
                detection['track_reactivated'] = True
            else:
                # Create new track
                track = VehicleTrack(vehicle_id, max_points=self.max_track_points)
                track.add_point(point)
                self.active_tracks[vehicle_id] = track
                detection['new_track'] = True
            
            # Add tracking info to detection
            detection['tracking_point'] = point.to_dict()
            detection['track_cameras'] = list(track.cameras_seen)
            detection['track_length'] = len(track.points)
            detection['track_duration'] = track.last_seen - track.first_seen
            
            updated_detections.append(detection)
            
        # Check for inactive tracks
        self._update_inactive_tracks(timestamp)
        
        # Limit number of stored tracks
        self._limit_tracks()
        
        return updated_detections
    
    def _register_camera_transition(self, from_camera: str, to_camera: str) -> None:
        """
        Register a transition between cameras
        
        Args:
            from_camera: Source camera ID
            to_camera: Destination camera ID
        """
        if from_camera == to_camera:
            return
            
        key = (from_camera, to_camera)
        self.camera_transitions[key] += 1
        
    def _update_inactive_tracks(self, current_time: float) -> None:
        """
        Move inactive tracks from active to inactive
        
        Args:
            current_time: Current timestamp
        """
        inactive_ids = []
        
        for vehicle_id, track in self.active_tracks.items():
            if current_time - track.last_seen > self.track_timeout:
                inactive_ids.append(vehicle_id)
                
        # Move inactive tracks
        for vehicle_id in inactive_ids:
            self.inactive_tracks[vehicle_id] = self.active_tracks.pop(vehicle_id)
    
    def _limit_tracks(self) -> None:
        """
        Limit the number of stored tracks
        """
        # First, make sure we don't exceed the total limit
        total_tracks = len(self.active_tracks) + len(self.inactive_tracks)
        
        if total_tracks <= self.max_tracks:
            return
            
        # Remove oldest inactive tracks until we're under the limit
        excess_count = total_tracks - self.max_tracks
        
        if excess_count > 0 and self.inactive_tracks:
            # Sort inactive tracks by last_seen (oldest first)
            sorted_tracks = sorted(
                self.inactive_tracks.items(), 
                key=lambda x: x[1].last_seen
            )
            
            # Remove oldest tracks
            for i in range(min(excess_count, len(sorted_tracks))):
                vehicle_id, _ = sorted_tracks[i]
                self.inactive_tracks.pop(vehicle_id, None)
    
    def get_track(self, vehicle_id: str) -> Optional[VehicleTrack]:
        """
        Get track for a specific vehicle
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            VehicleTrack or None if not found
        """
        return self.active_tracks.get(vehicle_id) or self.inactive_tracks.get(vehicle_id)
    
    def get_active_tracks(self) -> List[VehicleTrack]:
        """
        Get all active tracks
        
        Returns:
            List of active VehicleTracks
        """
        return list(self.active_tracks.values())
    
    def get_camera_statistics(self) -> Dict:
        """
        Get statistics about tracked vehicles per camera
        
        Returns:
            Dictionary with camera statistics
        """
        camera_counts = defaultdict(int)
        transition_counts = {}
        
        # Count vehicles seen per camera
        for track in self.active_tracks.values():
            for camera_id in track.cameras_seen:
                camera_counts[camera_id] += 1
                
        # Format transition counts
        for (from_cam, to_cam), count in self.camera_transitions.items():
            transition_counts[f"{from_cam}->{to_cam}"] = count
            
        return {
            'vehicles_per_camera': dict(camera_counts),
            'camera_transitions': transition_counts,
            'active_tracks': len(self.active_tracks),
            'inactive_tracks': len(self.inactive_tracks)
        }
    
    def draw_tracks(self, 
                   frame: np.ndarray,
                   camera_id: str,
                   max_trail_length: int = 20) -> np.ndarray:
        """
        Draw vehicle tracks on frame
        
        Args:
            frame: Input frame
            camera_id: Camera identifier
            max_trail_length: Maximum trail length to draw
            
        Returns:
            Frame with tracks drawn
        """
        img = frame.copy()
        
        # Draw tracks for active vehicles
        for vehicle_id, track in self.active_tracks.items():
            # Get points for this camera
            camera_points = track.get_points_by_camera(camera_id)
            
            if not camera_points:
                continue
                
            # Get trajectory (most recent points)
            trajectory = [p.center for p in camera_points[-max_trail_length:]]
            
            if len(trajectory) < 2:
                continue
                
            # Draw trajectory line
            color = self._get_id_color(vehicle_id)
            
            for i in range(1, len(trajectory)):
                pt1 = trajectory[i-1]
                pt2 = trajectory[i]
                thickness = int(5 * (i / len(trajectory))) + 1
                cv2.line(img, pt1, pt2, color, thickness)
                
            # Draw predicted position
            predicted_pos = track.predict_next_position(camera_id)
            if predicted_pos:
                cv2.circle(img, predicted_pos, 5, color, -1)
                
        return img
    
    def _get_id_color(self, vehicle_id: str) -> Tuple[int, int, int]:
        """
        Get consistent color based on vehicle ID
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            BGR color tuple
        """
        # Use hash of ID to generate consistent color
        hash_val = hash(vehicle_id) % (256**3)
        r = hash_val % 256
        g = (hash_val // 256) % 256
        b = (hash_val // (256**2)) % 256
        return (b, g, r)
    
    def save_tracks(self, filepath: str) -> None:
        """
        Save tracks to a file
        
        Args:
            filepath: Path to save file
        """
        # Combine active and inactive tracks
        all_tracks = {**self.active_tracks, **self.inactive_tracks}
        
        # Convert to serializable format
        data = {
            'tracks': {vid: track.to_dict() for vid, track in all_tracks.items()},
            'camera_transitions': {f"{a}->{b}": count for (a, b), count in self.camera_transitions.items()},
            'frame_counters': self.frame_counters,
            'timestamp': time.time()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_tracks(self, filepath: str) -> None:
        """
        Load tracks from a file
        
        Args:
            filepath: Path to load file
        """
        if not os.path.exists(filepath):
            print(f"Warning: Track file does not exist: {filepath}")
            return
            
        # Load from file
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Clear existing tracks
        self.active_tracks = {}
        self.inactive_tracks = {}
        
        # Load tracks
        for vehicle_id, track_data in data.get('tracks', {}).items():
            track = VehicleTrack.from_dict(track_data)
            
            # Determine if track is active or inactive
            current_time = time.time()
            if current_time - track.last_seen <= self.track_timeout:
                self.active_tracks[vehicle_id] = track
            else:
                self.inactive_tracks[vehicle_id] = track
                
        # Load camera transitions
        self.camera_transitions = defaultdict(int)
        for key_str, count in data.get('camera_transitions', {}).items():
            # Parse "camera1->camera2" format
            if '->' in key_str:
                from_cam, to_cam = key_str.split('->')
                self.camera_transitions[(from_cam, to_cam)] = count
                
        # Load frame counters
        self.frame_counters = defaultdict(int)
        for camera_id, count in data.get('frame_counters', {}).items():
            self.frame_counters[camera_id] = count


# Example usage
if __name__ == "__main__":
    import sys
    
    # Create vehicle identifier
    identifier = VehicleIdentifier()
    
    # Create multi-camera tracker
    tracker = MultiCameraTracker(identifier=identifier)
    
    # Example of processing frames from different cameras
    cameras = ['cam1', 'cam2']
    
    for camera_id in cameras:
        print(f"Processing video from camera {camera_id}")
        
        # Open a video
        video_path = f"path/to/{camera_id}_video.mp4"
        if not os.path.exists(video_path):
            print(f"Error: Video not found at {video_path}")
            # Use webcam as fallback
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_path)
        
        # Simple vehicle detector for demo
        detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        detector.classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 5 != 0:  # Process every 5th frame
                continue
                
            # Detect vehicles
            results = detector(frame)
            
            # Convert YOLOv5 detections to our format
            detections = []
            for pred in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = pred.cpu().numpy()
                
                detections.append({
                    'vehicle_bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf)
                })
                
            # Update tracker
            timestamp = time.time()
            tracked_detections = tracker.update(camera_id, detections, frame, timestamp)
            
            # Draw tracks
            frame_with_tracks = tracker.draw_tracks(frame, camera_id)
            
            # Display
            cv2.imshow(f'Tracking - {camera_id}', frame_with_tracks)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    # Save tracks
    tracker.save_tracks('vehicle_tracks.json')
    
    # Print statistics
    print("Camera Statistics:")
    stats = tracker.get_camera_statistics()
    print(json.dumps(stats, indent=2)) 