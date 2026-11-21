"""
Vehicle Classification and Identification Module
"""
import cv2
import numpy as np
import torch
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

# Local imports
from .. import config


class VehicleClassifier:
    """
    Classifies vehicles by type, color, make, and model
    """
    def __init__(self, 
                 model_path: str = None, 
                 device: str = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize vehicle classifier
        
        Args:
            model_path: Path to the classification model weights
            device: Device to run inference on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for classifications
        """
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"VehicleClassifier using device: {self.device}")
        
        # Load models
        self.type_model = self._load_type_model(model_path)
        self.color_model = self._load_color_model(model_path)
        self.make_model = self._load_make_model(model_path)
        
        # Define classes
        self.vehicle_types = [
            'sedan', 'suv', 'truck', 'van', 'pickup', 'bus', 
            'motorcycle', 'bicycle', 'train', 'boat'
        ]
        
        self.vehicle_colors = [
            'black', 'white', 'gray', 'silver', 'red', 'blue', 
            'green', 'yellow', 'brown', 'orange', 'purple', 'pink'
        ]
        
        self.vehicle_makes = [
            'toyota', 'honda', 'ford', 'chevrolet', 'bmw', 'mercedes',
            'audi', 'volkswagen', 'nissan', 'hyundai', 'kia', 'subaru',
            'lexus', 'mazda', 'jeep', 'ram', 'dodge', 'tesla', 'volvo'
        ]
        
    def _load_type_model(self, model_path: str):
        """
        Load the vehicle type classification model
        
        Args:
            model_path: Path to model weights
            
        Returns:
            The loaded model
        """
        # In a real implementation, we would load a trained model
        # For now, we'll use ResNet50 pretrained on ImageNet
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        model.to(self.device)
        model.eval()
        return model
    
    def _load_color_model(self, model_path: str):
        """
        Load the vehicle color classification model
        
        Args:
            model_path: Path to model weights
            
        Returns:
            The loaded model (placeholder for now)
        """
        # In a real implementation, this would be a separate model
        # For demo purposes, we'll use simulated results
        return None
    
    def _load_make_model(self, model_path: str):
        """
        Load the vehicle make classification model
        
        Args:
            model_path: Path to model weights
            
        Returns:
            The loaded model (placeholder for now)
        """
        # In a real implementation, this would be a separate model
        # For demo purposes, we'll use simulated results
        return None
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed tensor
        """
        # Resize
        img = cv2.resize(image, (224, 224))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize and convert to tensor
        img = img / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = torch.from_numpy(img).float()
        
        # Add batch dimension
        img = img.unsqueeze(0)
        
        return img
        
    def _predict_type(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict vehicle type
        
        Args:
            image: Vehicle image crop
            
        Returns:
            Tuple of (type, confidence)
        """
        # This is a simplified implementation
        # In a real system, we would use a model trained specifically for vehicle types
        
        # For demo purposes, return a simulated result
        import random
        vehicle_type = random.choice(self.vehicle_types)
        confidence = random.uniform(0.70, 0.95)
        
        return vehicle_type, confidence
    
    def _predict_color(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict vehicle color
        
        Args:
            image: Vehicle image crop
            
        Returns:
            Tuple of (color, confidence)
        """
        # For demo purposes, return a simulated result
        import random
        color = random.choice(self.vehicle_colors)
        confidence = random.uniform(0.65, 0.90)
        
        return color, confidence
    
    def _predict_make(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict vehicle make
        
        Args:
            image: Vehicle image crop
            
        Returns:
            Tuple of (make, confidence)
        """
        # For demo purposes, return a simulated result
        import random
        make = random.choice(self.vehicle_makes)
        confidence = random.uniform(0.60, 0.85)
        
        return make, confidence
    
    def classify(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Classify a vehicle
        
        Args:
            image: Full image
            bbox: Vehicle bounding box (x1, y1, x2, y2)
            
        Returns:
            Dictionary with classification results
        """
        x1, y1, x2, y2 = bbox
        
        # Extract vehicle region
        vehicle_img = image[y1:y2, x1:x2]
        
        if vehicle_img.size == 0:
            return {
                'vehicle_type': 'unknown',
                'type_confidence': 0.0,
                'color': 'unknown',
                'color_confidence': 0.0,
                'make': 'unknown',
                'make_confidence': 0.0
            }
        
        # Get predictions
        vehicle_type, type_conf = self._predict_type(vehicle_img)
        color, color_conf = self._predict_color(vehicle_img)
        make, make_conf = self._predict_make(vehicle_img)
        
        return {
            'vehicle_type': vehicle_type,
            'type_confidence': type_conf,
            'color': color,
            'color_confidence': color_conf,
            'make': make,
            'make_confidence': make_conf
        }


class VehicleIdentifier:
    """
    Identifies unique vehicles based on appearance and tracks them across frames
    """
    def __init__(self, 
                 feature_extractor_path: str = None,
                 similarity_threshold: float = 0.7,
                 device: str = None):
        """
        Initialize vehicle identifier
        
        Args:
            feature_extractor_path: Path to feature extractor model
            similarity_threshold: Threshold for considering two vehicles the same
            device: Device to run inference on ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"VehicleIdentifier using device: {self.device}")
        
        self.similarity_threshold = similarity_threshold
        self.feature_extractor = self._load_feature_extractor(feature_extractor_path)
        
        # Store vehicle identities
        self.vehicle_features = {}  # vehicle_id -> feature vector
        self.vehicle_metadata = {}  # vehicle_id -> metadata
        
    def _load_feature_extractor(self, model_path: str):
        """
        Load the feature extractor model
        
        Args:
            model_path: Path to model weights
            
        Returns:
            The loaded model
        """
        # In a real implementation, we would load a specialized model
        # For demo purposes, we'll use ResNet as a feature extractor
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        
        # Remove the classification layer to use as feature extractor
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.to(self.device)
        model.eval()
        
        return model
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from vehicle image
        
        Args:
            image: Vehicle image crop
            
        Returns:
            Feature vector
        """
        # Resize
        img = cv2.resize(image, (224, 224))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize and convert to tensor
        img = img / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = torch.from_numpy(img).float()
        
        # Add batch dimension
        img = img.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(img)
            
        # Flatten and normalize
        features = features.squeeze().cpu().numpy()
        features = features / np.linalg.norm(features)
        
        return features
    
    def _compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute similarity between two feature vectors
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Similarity score (0-1)
        """
        return np.dot(features1, features2)
    
    def _find_matching_vehicle(self, features: np.ndarray) -> Optional[str]:
        """
        Find a matching vehicle in the database
        
        Args:
            features: Feature vector
            
        Returns:
            Vehicle ID if match found, None otherwise
        """
        best_match = None
        best_score = 0
        
        for vehicle_id, stored_features in self.vehicle_features.items():
            similarity = self._compute_similarity(features, stored_features)
            
            if similarity > self.similarity_threshold and similarity > best_score:
                best_match = vehicle_id
                best_score = similarity
                
        return best_match
    
    def identify(self, 
                 image: np.ndarray, 
                 bbox: Tuple[int, int, int, int],
                 metadata: Dict) -> Tuple[str, bool]:
        """
        Identify a vehicle
        
        Args:
            image: Full image
            bbox: Vehicle bounding box (x1, y1, x2, y2)
            metadata: Additional vehicle information
            
        Returns:
            Tuple of (vehicle_id, is_new_vehicle)
        """
        x1, y1, x2, y2 = bbox
        
        # Extract vehicle region
        vehicle_img = image[y1:y2, x1:x2]
        
        if vehicle_img.size == 0:
            return str(uuid.uuid4()), True
            
        # Extract features
        features = self.extract_features(vehicle_img)
        
        # Find matching vehicle
        vehicle_id = self._find_matching_vehicle(features)
        
        # If no match found, create new identity
        if vehicle_id is None:
            vehicle_id = str(uuid.uuid4())
            self.vehicle_features[vehicle_id] = features
            self.vehicle_metadata[vehicle_id] = {
                'first_seen': time.time(),
                **metadata
            }
            return vehicle_id, True
        else:
            # Update metadata for existing vehicle
            self.vehicle_metadata[vehicle_id].update({
                'last_seen': time.time(),
                **metadata
            })
            return vehicle_id, False
    
    def get_vehicle_count(self) -> int:
        """
        Get the number of unique vehicles identified
        
        Returns:
            Number of unique vehicles
        """
        return len(self.vehicle_features)
    
    def get_vehicle_metadata(self, vehicle_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific vehicle
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            Vehicle metadata or None if not found
        """
        return self.vehicle_metadata.get(vehicle_id)
    
    def get_all_vehicles(self) -> Dict[str, Dict]:
        """
        Get all vehicle metadata
        
        Returns:
            Dictionary mapping vehicle IDs to metadata
        """
        return self.vehicle_metadata


class VehicleClassificationSystem:
    """
    Complete vehicle classification and identification system
    """
    def __init__(self,
                 classifier_model_path: str = None,
                 identifier_model_path: str = None,
                 confidence_threshold: float = 0.5,
                 similarity_threshold: float = 0.7,
                 device: str = None):
        """
        Initialize vehicle classification system
        
        Args:
            classifier_model_path: Path to classifier model
            identifier_model_path: Path to identifier model
            confidence_threshold: Minimum confidence for classifications
            similarity_threshold: Threshold for vehicle identity matching
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.classifier = VehicleClassifier(
            model_path=classifier_model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )
        
        self.identifier = VehicleIdentifier(
            feature_extractor_path=identifier_model_path,
            similarity_threshold=similarity_threshold,
            device=device
        )
        
        # Track detections by ID
        self.tracked_vehicles = {}
        
    def process_detections(self, 
                          image: np.ndarray, 
                          detections: List[Dict]) -> List[Dict]:
        """
        Process vehicle detections
        
        Args:
            image: Input image
            detections: List of vehicle detections with bounding boxes
            
        Returns:
            List of detections with classification and identification results
        """
        results = []
        
        for detection in detections:
            # Extract bounding box
            bbox = detection.get('vehicle_bbox') or detection.get('bbox')
            
            if not bbox:
                continue
                
            # Classify vehicle
            classification = self.classifier.classify(image, bbox)
            
            # Identify vehicle
            metadata = {**detection, **classification}
            vehicle_id, is_new = self.identifier.identify(image, bbox, metadata)
            
            # Create result
            result = {
                **detection,
                **classification,
                'vehicle_id': vehicle_id,
                'is_new_vehicle': is_new,
                'timestamp': time.time()
            }
            
            # Update tracked vehicles
            self.tracked_vehicles[vehicle_id] = result
            
            results.append(result)
            
        return results
        
    def draw_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        Draw detection results on an image
        
        Args:
            frame: Input image
            results: Results from process_detections
            
        Returns:
            Image with detections drawn
        """
        img = frame.copy()
        
        for result in results:
            # Draw bounding box
            bbox = result.get('vehicle_bbox') or result.get('bbox')
            
            if not bbox:
                continue
                
            x1, y1, x2, y2 = bbox
            
            # Draw box with color based on vehicle type
            color_map = {
                'sedan': (0, 255, 0),
                'suv': (0, 200, 100),
                'truck': (0, 100, 200),
                'van': (100, 200, 0),
                'pickup': (200, 100, 0),
                'bus': (200, 0, 100),
                'motorcycle': (100, 0, 200)
            }
            
            box_color = color_map.get(result.get('vehicle_type', 'unknown'), (0, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw vehicle ID
            vehicle_id = result.get('vehicle_id', '')
            short_id = vehicle_id[:8] if vehicle_id else ''
            
            cv2.putText(
                img, 
                f"ID: {short_id}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2
            )
            
            # Draw vehicle info
            v_type = result.get('vehicle_type', 'unknown')
            color = result.get('color', 'unknown')
            make = result.get('make', 'unknown')
            
            cv2.putText(
                img, 
                f"{color} {make} {v_type}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2
            )
                
        return img
    
    def get_statistics(self) -> Dict:
        """
        Get system statistics
        
        Returns:
            Dictionary with statistics
        """
        vehicle_count = self.identifier.get_vehicle_count()
        
        # Count by type and color
        types_count = defaultdict(int)
        colors_count = defaultdict(int)
        makes_count = defaultdict(int)
        
        for vehicle_id, data in self.tracked_vehicles.items():
            v_type = data.get('vehicle_type', 'unknown')
            color = data.get('color', 'unknown')
            make = data.get('make', 'unknown')
            
            types_count[v_type] += 1
            colors_count[color] += 1
            makes_count[make] += 1
            
        return {
            'total_vehicles': vehicle_count,
            'types_count': dict(types_count),
            'colors_count': dict(colors_count),
            'makes_count': dict(makes_count)
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    # Create vehicle classification system
    system = VehicleClassificationSystem()
    
    # Open a video
    video_path = "path/to/video.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        # Use webcam as fallback
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    
    # Simple vehicle detector for demo
    detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    detector.classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
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
            
        # Process detections
        classification_results = system.process_detections(frame, detections)
        
        # Draw results
        frame_with_results = system.draw_results(frame, classification_results)
        
        # Display
        cv2.imshow('Vehicle Classification', frame_with_results)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows() 