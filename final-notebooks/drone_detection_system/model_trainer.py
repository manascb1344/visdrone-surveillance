import os
import shutil
from ultralytics import YOLO
import torch

class ModelTrainer:
    
    def __init__(self, project_dir="./visdrone_yolo_training"):
        self.project_dir = project_dir
        os.makedirs(project_dir, exist_ok=True)
    
    def train_yolo(self, 
                  dataset_yaml, 
                  model_size="n", 
                  epochs=50, 
                  batch_size=16, 
                  img_size=640, 
                  num_gpus=1):
        if num_gpus <= 0:
            print("Number of GPUs must be positive. Defaulting to 1 GPU.")
            device_str = '0'
            run_name = f"yolo11{model_size}_visdrone_1GPU"
            effective_batch_per_gpu = batch_size
        elif num_gpus == 1:
             device_str = '0'
             run_name = f"yolo11{model_size}_visdrone_1GPU"
             effective_batch_per_gpu = batch_size
        else:
            device_str = ','.join(map(str, range(num_gpus)))
            run_name = f"yolo11{model_size}_visdrone_{num_gpus}GPU"
            effective_batch_per_gpu = batch_size // num_gpus

        print(f"Training yolo11{model_size} on VisDrone dataset for {epochs} epochs using {num_gpus} GPU(s) ({device_str}).")
        print(f"Total batch size: {batch_size} (Effective batch size per GPU: {effective_batch_per_gpu})")
        if batch_size % num_gpus != 0 and num_gpus > 1:
             print(f"Warning: Batch size {batch_size} is not perfectly divisible by {num_gpus} GPUs.")

        model = YOLO(f"yolo11{model_size}.pt")

        try:
            num_workers = min(16, os.cpu_count()) 
        except NotImplementedError:
            num_workers = 8 
        print(f"Using {num_workers} data loading workers.")

        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,        
            imgsz=img_size,
            patience=20,             
            save=True,               
            device=device_str,       
            project=self.project_dir,
            name=run_name,           
            exist_ok=True,
            pretrained=True,
            optimizer="Adam",
            workers=num_workers,     
        )

        print("Training completed.")
        
        training_run_dir = os.path.join(self.project_dir, run_name)
        best_model_path = os.path.join(training_run_dir, "weights", "best.pt")
        
        return {
            "model": model,
            "results": results,
            "best_model_path": best_model_path,
            "run_dir": training_run_dir
        }
    
    def save_model(self, model_path, save_dir, new_name=None):
        os.makedirs(save_dir, exist_ok=True)
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
            
        if new_name:
            target_path = os.path.join(save_dir, new_name)
        else:
            target_path = os.path.join(save_dir, os.path.basename(model_path))
            
        try:
            shutil.copy(model_path, target_path)
            print(f"Model saved to: {target_path}")
            return target_path
        except Exception as e:
            print(f"Error saving model: {e}")
            return None 