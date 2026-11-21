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
        available_gpus = torch.cuda.device_count()
        
        if available_gpus == 0:
            device_str = 'cpu'
            run_name = f"yolo11{model_size}_visdrone_CPU"
        elif num_gpus <= 0:
            device_str = '0'
            run_name = f"yolo11{model_size}_visdrone_1GPU"
        elif num_gpus > available_gpus:
            num_gpus = available_gpus
            device_str = ','.join(map(str, range(num_gpus)))
            run_name = f"yolo11{model_size}_visdrone_{num_gpus}GPU"
        elif num_gpus == 1:
            device_str = '0'
            run_name = f"yolo11{model_size}_visdrone_1GPU"
        else:
            device_str = ','.join(map(str, range(num_gpus)))
            run_name = f"yolo11{model_size}_visdrone_{num_gpus}GPU"

        model = YOLO(f"yolo11{model_size}.pt")

        try:
            num_workers = min(16, os.cpu_count()) 
        except NotImplementedError:
            num_workers = 8 

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
            return None
            
        if new_name:
            target_path = os.path.join(save_dir, new_name)
        else:
            target_path = os.path.join(save_dir, os.path.basename(model_path))
            
        try:
            shutil.copy(model_path, target_path)
            return target_path
        except Exception as e:
            return None 