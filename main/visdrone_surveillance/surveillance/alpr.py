"""
Automated License Plate Recognition (ALPR) Module
"""
import cv2
import numpy as np
import torch
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Local imports
from .. import config


class LicensePlateDetector:
    """
    Detects license plates in images using a two-stage approach:
    1. Detect vehicles
    2. Locate license plates within vehicle bounding boxes
    """
    def __init__(self, 
                 model_path: str = None,
                 confidence_threshold: float = 0.5,
                 device: str = None):
        """
        Initialize license plate detector
        
        Args:
            model_path: Path to the detection model weights
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"LicensePlateDetector using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        """
        Load the license plate detection model
        
        Args:
            model_path: Path to model weights
            
        Returns:
            The loaded model
        """
        if model_path is None:
            # Use a pre-trained YOLOv5 model for vehicle detection
            # In a real implementation, we would fine-tune this model on license plate data
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        else:
            # Load custom model
            model = torch.load(model_path, map_location=self.device)
            
        model.to(self.device)
        model.eval()
        return model
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect license plates in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detections with format:
            [
                {
                    'vehicle_bbox': (x1, y1, x2, y2),
                    'vehicle_type': str,
                    'vehicle_confidence': float,
                    'plate_bbox': (x1, y1, x2, y2),
                    'plate_confidence': float
                },
                ...
            ]
        """
        # Get vehicle detections
        results = self.model(image)
        detections = []
        
        # Process results
        for pred in results.xyxy[0]:
            # Get detection data
            x1, y1, x2, y2, conf, cls = pred.cpu().numpy()
            
            if conf >= self.confidence_threshold:
                # For now, this is a placeholder implementation
                # In a full implementation, we would:
                # 1. Get the vehicle crop from the image
                # 2. Run a license plate detector on the crop
                # 3. Return both vehicle and plate information
                
                # Simulate plate detection for now
                vehicle_width = x2 - x1
                vehicle_height = y2 - y1
                
                # Roughly estimate where a license plate might be
                plate_width = vehicle_width * 0.4
                plate_height = vehicle_height * 0.1
                plate_x1 = x1 + (vehicle_width - plate_width) / 2
                plate_y1 = y1 + vehicle_height * 0.7  # Plates are usually in the lower part
                plate_x2 = plate_x1 + plate_width
                plate_y2 = plate_y1 + plate_height
                
                class_mapping = {
                    2: 'car',
                    3: 'motorcycle',
                    5: 'bus',
                    7: 'truck'
                }
                
                detections.append({
                    'vehicle_bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'vehicle_type': class_mapping.get(int(cls), 'vehicle'),
                    'vehicle_confidence': float(conf),
                    'plate_bbox': (int(plate_x1), int(plate_y1), int(plate_x2), int(plate_y2)),
                    'plate_confidence': float(conf) * 0.8  # Simulated confidence
                })
                
        return detections


class LicensePlateRecognizer:
    """
    Recognizes text on license plates using OCR
    """
    def __init__(self, ocr_model_path: str = None):
        """
        Initialize license plate recognizer
        
        Args:
            ocr_model_path: Path to OCR model (optional)
        """
        # In a real implementation, we would load a proper OCR model
        # This is a placeholder that will be implemented with a real OCR engine
        self.initialized = True
        print("License Plate Recognizer initialized")
        
    def recognize(self, 
                  image: np.ndarray, 
                  plate_bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        Recognize text on a license plate
        
        Args:
            image: Input image (BGR format)
            plate_bbox: License plate bounding box (x1, y1, x2, y2)
            
        Returns:
            Tuple of (plate_text, confidence)
        """
        x1, y1, x2, y2 = plate_bbox
        
        # Extract plate region
        plate_img = image[y1:y2, x1:x2]
        
        # Preprocess image
        if plate_img.size == 0:
            return "", 0.0
            
        # Resize for better OCR
        plate_img = cv2.resize(plate_img, (int((x2-x1)*2), int((y2-y1)*2)))
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # In a real implementation, we would:
        # 1. Use a trained OCR model (like Tesseract or a custom CNN)
        # 2. Extract the text from the plate image
        # 3. Apply post-processing to validate the plate format
        
        # This is a placeholder that generates a random plate for demonstration
        plate_formats = [
            "ABC123",
            "AB123CD",
            "123XYZ",
            "XY12345"
        ]
        import random
        plate_text = random.choice(plate_formats)
        confidence = random.uniform(0.70, 0.95)
        
        return plate_text, confidence


class ALPR:
    """
    Complete Automated License Plate Recognition system
    """
    def __init__(self, 
                 detector_model_path: str = None,
                 recognizer_model_path: str = None,
                 confidence_threshold: float = 0.5,
                 device: str = None):
        """
        Initialize ALPR system
        
        Args:
            detector_model_path: Path to detection model
            recognizer_model_path: Path to recognition model
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.detector = LicensePlateDetector(
            model_path=detector_model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )
        
        self.recognizer = LicensePlateRecognizer(
            ocr_model_path=recognizer_model_path
        )
        
        self.results_history = []
        
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a single frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of results with format:
            [
                {
                    'vehicle_bbox': (x1, y1, x2, y2),
                    'vehicle_type': str,
                    'vehicle_confidence': float,
                    'plate_bbox': (x1, y1, x2, y2),
                    'plate_confidence': float,
                    'plate_text': str,
                    'text_confidence': float,
                    'timestamp': float
                },
                ...
            ]
        """
        # Detect license plates
        detections = self.detector.detect(frame)
        results = []
        
        # Recognize text on each plate
        for detection in detections:
            plate_text, text_conf = self.recognizer.recognize(
                frame, detection['plate_bbox']
            )
            
            result = {
                **detection,
                'plate_text': plate_text,
                'text_confidence': text_conf,
                'timestamp': time.time()
            }
            
            results.append(result)
            
        # Store results for later analysis
        self.results_history.extend(results)
        if len(self.results_history) > 1000:  # Keep last 1000 results
            self.results_history = self.results_history[-1000:]
            
        return results
        
    def draw_results(self, 
                     frame: np.ndarray, 
                     results: List[Dict],
                     draw_vehicles: bool = True,
                     draw_plates: bool = True) -> np.ndarray:
        """
        Draw detection results on an image
        
        Args:
            frame: Input image
            results: Results from process_frame
            draw_vehicles: Whether to draw vehicle bounding boxes
            draw_plates: Whether to draw license plate bounding boxes
            
        Returns:
            Image with detections drawn
        """
        img = frame.copy()
        
        for result in results:
            # Draw vehicle bbox
            if draw_vehicles:
                x1, y1, x2, y2 = result['vehicle_bbox']
                vehicle_type = result['vehicle_type']
                conf = result['vehicle_confidence']
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img, 
                    f"{vehicle_type} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            # Draw plate bbox and text
            if draw_plates:
                x1, y1, x2, y2 = result['plate_bbox']
                plate_text = result['plate_text']
                text_conf = result['text_confidence']
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img, 
                    f"{plate_text} {text_conf:.2f}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
                
        return img
    
    def check_plate_against_database(self, plate_text: str) -> Dict:
        """
        Check a license plate against a database of interest
        
        Args:
            plate_text: The license plate text to check
            
        Returns:
            Dictionary with check results
        """
        # This is a placeholder for database checking functionality
        # In a real implementation, this would connect to a database
        
        # Simulate database check
        import random
        is_match = random.random() < 0.05  # 5% chance of match
        
        if is_match:
            return {
                'match': True,
                'category': random.choice(['stolen', 'wanted', 'expired']),
                'confidence': random.uniform(0.8, 0.99),
                'plate': plate_text
            }
        else:
            return {
                'match': False,
                'plate': plate_text
            }
    
    def get_plate_statistics(self) -> Dict:
        """
        Get statistics on processed plates
        
        Returns:
            Dictionary with statistics
        """
        if not self.results_history:
            return {'total_plates': 0}
            
        plates = [r['plate_text'] for r in self.results_history]
        unique_plates = set(plates)
        
        vehicle_types = {}
        for r in self.results_history:
            v_type = r['vehicle_type']
            if v_type in vehicle_types:
                vehicle_types[v_type] += 1
            else:
                vehicle_types[v_type] = 1
                
        return {
            'total_plates': len(plates),
            'unique_plates': len(unique_plates),
            'vehicle_types': vehicle_types
        }


# Example usage
if __name__ == "__main__":
    # Create ALPR system
    alpr = ALPR()
    
    # Open a video
    video_path = "path/to/video.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        # Use webcam as fallback
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        results = alpr.process_frame(frame)
        
        # Draw results
        frame_with_detections = alpr.draw_results(frame, results)
        
        # Display
        cv2.imshow('ALPR', frame_with_detections)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows() 