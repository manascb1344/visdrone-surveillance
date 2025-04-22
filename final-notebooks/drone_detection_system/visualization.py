import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from IPython.display import display, Image
import itertools

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

        self.base_colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 165, 0),
            (128, 0, 128),
            (0, 128, 128),
            (128, 128, 0),
        ]

    def visualize_training_results(self, results_path):
        """
        Display training results from a CSV file.

        Args:
            results_path (str): Path to the results.csv file
        """
        try:
            results = pd.read_csv(results_path)

            plt.figure(figsize=(20, 10))

            plt.subplot(2, 2, 1)
            plt.plot(results['epoch'], results['train/box_loss'], label='train')
            plt.plot(results['epoch'], results['val/box_loss'], label='val')
            plt.title('Box Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(results['epoch'], results['metrics/mAP50(B)'], label='mAP50')
            plt.plot(results['epoch'], results['metrics/mAP50-95(B)'], label='mAP50-95')
            plt.title('mAP Metrics')
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.legend()

            plt.subplot(2, 2, 3)
            plt.plot(results['epoch'], results['metrics/precision(B)'], label='precision')
            plt.plot(results['epoch'], results['metrics/recall(B)'], label='recall')
            plt.title('Precision & Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()

            plt.tight_layout()
            plt.show()

            conf_matrix_path = os.path.dirname(results_path) + "/confusion_matrix.png"
            if os.path.exists(conf_matrix_path):
                display(Image(filename=conf_matrix_path))

            return True
        except Exception as e:
            print(f"Error visualizing results: {e}")
            return False

    def test_model_on_samples(self, model_path, sample_dir, num_samples=5, conf_threshold=0.25):
        """
        Test one or more trained models on sample images (ensemble visualization).

        Args:
            model_path (list[str]): List of paths to the trained models.
            sample_dir (str): Directory containing sample images.
            num_samples (int): Number of samples to test.
            conf_threshold (float): Confidence threshold for detection.

        Returns:
            bool: Success status.
        """
        if not isinstance(model_path, list):
            model_path = [model_path]

        print(f"Testing {len(model_path)} model(s) on {num_samples} sample images...")
        print("Model paths:", model_path)

        models = []
        model_names = []
        for model_path in model_path:
            try:
                models.append(YOLO(model_path))
                model_names.append(os.path.splitext(os.path.basename(model_path))[0])
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")

        if not models:
            print("No models were loaded successfully.")
            return False

        ensemble_colors = list(itertools.islice(itertools.cycle(self.base_colors), len(models)))

        all_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            try:
                all_images.extend([f for f in os.listdir(sample_dir)
                                   if os.path.isfile(os.path.join(sample_dir, f)) and
                                      f.lower().endswith(ext.replace('*', ''))])
            except FileNotFoundError:
                 print(f"Error: Sample directory not found: {sample_dir}")
                 return False

        if not all_images:
            print(f"No images found in {sample_dir}")
            return False

        if len(all_images) > num_samples:
            sample_images = random.sample(all_images, num_samples)
        else:
            sample_images = all_images

        num_images_to_show = len(sample_images)
        plt.figure(figsize=(20, 6 * num_images_to_show))

        for i, img_name in enumerate(sample_images):
            img_path = os.path.join(sample_dir, img_name)
            print(f"Processing: {img_path}")

            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  Warning: Could not read image {img_path}. Skipping.")
                    continue
                vis_img = img.copy()
            except Exception as e:
                print(f"  Error loading image {img_path}: {e}. Skipping.")
                continue

            for model_idx, model in enumerate(models):
                model_name = model_names[model_idx]
                color = ensemble_colors[model_idx]

                try:
                    results = model(img_path, conf=conf_threshold, verbose=False)
                    boxes = results[0].boxes

                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        x1, y1, x2, y2 = xyxy

                        class_name = self.class_names[cls] if cls < len(self.class_names) else f"CLS {cls}"
                        label = f'{model_name}: {class_name} {conf:.2f}'

                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

                        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(vis_img, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, -1)
                        cv2.putText(vis_img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

                except Exception as e:
                    print(f"  Error during inference with model {model_name} on {img_name}: {e}")

            plt.subplot(num_images_to_show, 1, i + 1)
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Ensemble Result: {img_name}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        return True