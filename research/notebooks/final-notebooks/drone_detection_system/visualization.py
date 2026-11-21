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
    def __init__(self, class_names=None):
        if class_names is None:
            self.class_names = [
                'person', 'bicycle', 'car', 'van',
                'truck', 'tricycle', 'bus', 'motorcycle'
            ]
        else:
            self.class_names = class_names

        # Base colors for visualization
        self.base_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]

    def visualize_training_results(self, results_path):
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
        if not isinstance(model_path, list):
            model_path = [model_path]

        print(f"Testing {len(model_path)} model(s) on {num_samples} sample images...")
        print("Model paths:", model_path)

        models = []
        model_names = []
        for path in model_path:
            try:
                models.append(YOLO(path))
                model_names.append(os.path.splitext(os.path.basename(path))[0])
            except Exception as e:
                print(f"Error loading model {path}: {e}")

        if not models:
            print("No models were loaded successfully.")
            return False

        ensemble_colors = []
        color_cycle = itertools.cycle(self.base_colors)
        for _ in range(len(models)):
            ensemble_colors.append(next(color_cycle))

        all_images = self._get_images_from_directory(sample_dir)
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

                self._process_image_with_model(img_path, vis_img, model, model_name, color, conf_threshold)

            plt.subplot(num_images_to_show, 1, i + 1)
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Ensemble Result: {img_name}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        return True
    def _get_images_from_directory(self, directory):
        all_images = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        
        for ext in image_extensions:
            try:
                clean_ext = ext.replace('*', '')
                
                for f in os.listdir(directory):
                    file_path = os.path.join(directory, f)
                    if os.path.isfile(file_path) and f.lower().endswith(clean_ext):
                        all_images.append(f)
            except FileNotFoundError:
                print(f"Error: Sample directory not found: {directory}")
                return []
                
        return all_images
    
    def _process_image_with_model(self, img_path, vis_img, model, model_name, color, conf_threshold):
        try:
            results = model(img_path, conf=conf_threshold, verbose=False)
            boxes = results[0].boxes
            
            if not hasattr(boxes, 'xyxy') or len(boxes.xyxy) == 0:
                print(f"  Model {model_name} found no boxes above threshold {conf_threshold}")
                return
                
            print(f"  Model {model_name} detected {len(boxes.xyxy)} boxes")
            
            for j in range(len(boxes.xyxy)):
                try:
                    xyxy = boxes.xyxy[j].cpu().numpy().astype(int)
                    
                    if hasattr(boxes, 'conf'):
                        conf = boxes.conf[j].cpu().numpy()
                    else:
                        conf = 0.0
                    
                    if hasattr(boxes, 'cls'):
                        cls = int(boxes.cls[j].cpu().numpy())
                    else:
                        cls = 0
                    
                    x1, y1, x2, y2 = xyxy
                    
                    if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
                        if cls < len(self.class_names):
                            class_name = self.class_names[cls]
                        else:
                            class_name = f"CLS {cls}"
                            
                        label = f'{model_name}: {class_name} {conf:.2f}'
                        
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                        
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        label_width = text_size[0][0]
                        label_height = text_size[0][1]
                        baseline = text_size[1]
                        
                        cv2.rectangle(
                            vis_img, 
                            (x1, y1 - label_height - baseline), 
                            (x1 + label_width, y1), 
                            color, 
                            -1
                        )
                        
                        cv2.putText(
                            vis_img, 
                            label, 
                            (x1, y1 - baseline), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 0, 0), 
                            1
                        )
                except Exception as e:
                    print(f"  Error processing box {j}: {e}")
                    
        except Exception as e:
            print(f"  Error during inference with model {model_name}: {e}")