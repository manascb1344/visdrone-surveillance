import os
import shutil
from ultralytics import YOLO
import torch

class ModelTrainer:
    """
    Class for training yolo11 models on the VisDrone dataset.
    """
    
    def __init__(self, project_dir="./visdrone_yolo_training"):
        """
        Initialize the model trainer.
        
        Args:
            project_dir (str): Directory to save training results
        """
        self.project_dir = project_dir
        os.makedirs(project_dir, exist_ok=True)
    
    def train_yolo(self, 
                  dataset_yaml, 
                  model_size="n", 
                  epochs=50, 
                  batch_size=16, 
                  img_size=640, 
                  num_gpus=1):
        """
        Train yolo11 on the converted VisDrone dataset using multiple GPUs.

        Args:
            dataset_yaml (str): Path to the dataset YAML file
            model_size (str): YOLO model size (n, s, m, l, x)
            epochs (int): Number of training epochs
            batch_size (int): Total batch size across all GPUs
            img_size (int): Image size for training
            num_gpus (int): Number of GPUs to use
            
        Returns:
            object: Training results
        """
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
            # Create device string like '0,1' for num_gpus=2
            device_str = ','.join(map(str, range(num_gpus)))
            run_name = f"yolo11{model_size}_visdrone_{num_gpus}GPU"
            effective_batch_per_gpu = batch_size // num_gpus

        print(f"Training yolo11{model_size} on VisDrone dataset for {epochs} epochs using {num_gpus} GPU(s) ({device_str}).")
        print(f"Total batch size: {batch_size} (Effective batch size per GPU: {effective_batch_per_gpu})")
        if batch_size % num_gpus != 0 and num_gpus > 1:
             print(f"Warning: Batch size {batch_size} is not perfectly divisible by {num_gpus} GPUs.")

        # Load pre-trained yolo11 model
        model = YOLO(f"yolo11{model_size}.pt")

        # Define number of workers for data loading
        try:
            num_workers = min(16, os.cpu_count()) # Use up to 16 workers or max available cores
        except NotImplementedError:
            num_workers = 8 # Default if cpu_count() is not available
        print(f"Using {num_workers} data loading workers.")

        # Train the model using specified GPUs
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,        # Total batch size across all devices
            imgsz=img_size,
            patience=20,             # Early stopping patience
            save=True,               # Save checkpoints
            device=device_str,       # Specify the GPUs to use
            project=self.project_dir,
            name=run_name,           # Use the modified run name for output folder
            exist_ok=True,
            pretrained=True,
            optimizer="Adam",
            workers=num_workers,     # Set number of data loader workers
        )

        print("Training completed.")
        
        # Get path to best model
        training_run_dir = os.path.join(self.project_dir, run_name)
        best_model_path = os.path.join(training_run_dir, "weights", "best.pt")
        
        return {
            "model": model,
            "results": results,
            "best_model_path": best_model_path,
            "run_dir": training_run_dir
        }
    
    def save_model(self, model_path, save_dir, new_name=None):
        """
        Save a copy of the model to a new location.
        
        Args:
            model_path (str): Path to the model to save
            save_dir (str): Directory to save the model to
            new_name (str, optional): New name for the model
            
        Returns:
            str: Path to the saved model
        """
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