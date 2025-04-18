import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from IPython.display import display, Image

class Visualizer:
    """
    Class for visualizing training results and model predictions.
    """
    
    def __init__(self, class_names=None):
        """
        Initialize the visualizer.
        
        Args:
            class_names (list, optional): List of class names
        """
        self.class_names = class_names or [
            'person', 'bicycle', 'car', 'van', 
            'truck', 'tricycle', 'bus', 'motorcycle'
        ]
        
        # Define colors for visualization
        self.colors = [
            (0, 255, 0),    # person: green
            (255, 0, 0),    # bicycle: blue
            (0, 0, 255),    # car: red
            (255, 255, 0),  # van: cyan
            (255, 0, 255),  # truck: magenta
            (0, 255, 255),  # tricycle: yellow
            (255, 165, 0),  # bus: orange
            (128, 0, 128)   # motorcycle: purple
        ]
    
    def visualize_training_results(self, results_path):
        """
        Display training results from a CSV file.
        
        Args:
            results_path (str): Path to the results.csv file
        """
        try:
            # Display training curves
            results = pd.read_csv(results_path)

            plt.figure(figsize=(20, 10))

            # Plot training loss
            plt.subplot(2, 2, 1)
            plt.plot(results['epoch'], results['train/box_loss'], label='train')
            plt.plot(results['epoch'], results['val/box_loss'], label='val')
            plt.title('Box Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # Plot mAP metrics
            plt.subplot(2, 2, 2)
            plt.plot(results['epoch'], results['metrics/mAP50(B)'], label='mAP50')
            plt.plot(results['epoch'], results['metrics/mAP50-95(B)'], label='mAP50-95')
            plt.title('mAP Metrics')
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.legend()

            # Plot precision and recall
            plt.subplot(2, 2, 3)
            plt.plot(results['epoch'], results['metrics/precision(B)'], label='precision')
            plt.plot(results['epoch'], results['metrics/recall(B)'], label='recall')
            plt.title('Precision & Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()

            plt.tight_layout()
            plt.show()

            # Show confusion matrix
            conf_matrix_path = os.path.dirname(results_path) + "/confusion_matrix.png"
            if os.path.exists(conf_matrix_path):
                display(Image(filename=conf_matrix_path))
            
            return True
        except Exception as e:
            print(f"Error visualizing results: {e}")
            return False
    
    def test_model_on_samples(self, model_path, sample_dir, num_samples=5, conf_threshold=0.25):
        """
        Test a trained model on sample images.
        
        Args:
            model_path (str): Path to the trained model
            sample_dir (str): Directory containing sample images
            num_samples (int): Number of samples to test
            conf_threshold (float): Confidence threshold for detection
            
        Returns:
            bool: Success status
        """
        print(f"Testing model on {num_samples} sample images...")

        # Load the model
        try:
            model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

        # Get sample images
        all_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            all_images.extend([f for f in os.listdir(sample_dir) 
                              if f.lower().endswith(ext.replace('*', ''))])
        
        if not all_images:
            print(f"No images found in {sample_dir}")
            return False
            
        if len(all_images) > num_samples:
            sample_images = random.sample(all_images, num_samples)
        else:
            sample_images = all_images

        # Predict and display results
        plt.figure(figsize=(20, 4*num_samples))

        for i, img_name in enumerate(sample_images):
            img_path = os.path.join(sample_dir, img_name)
            
            # Run prediction
            results = model(img_path, conf=conf_threshold)

            # Get the result image with annotations
            result_img = results[0].plot()

            # Display the result
            plt.subplot(num_samples, 1, i+1)
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Sample {i+1}: {img_name}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        
        return True 