#!/usr/bin/env python3
import os
import argparse
import torch
import gc

from data_manager import DataManager
from model_trainer import ModelTrainer
from visualization import Visualizer
from rtmp_processor import RTMPProcessor
from ultralytics import YOLO
from faster_rcnn import FasterRCNNTrainer, FasterRCNNWrapper

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description='Drone Detection System')
    
    parser.add_argument('--mode', type=str, choices=['prepare', 'train', 'test', 'rtmp', 'export'],
                       default='train', help='Operation mode')
    
    parser.add_argument('--base-path', type=str, default='.',
                       help='Base path for dataset storage')
    parser.add_argument('--project-dir', type=str, default='./visdrone_yolo_training',
                       help='Directory to save training results')
    parser.add_argument('--rtmp-url', type=str, default='rtmp://192.168.1.1/live/stream',
                       help='RTMP URL for live processing')
    
    parser.add_argument('--model-type', type=str, default='yolo', choices=['yolo', 'faster_rcnn'],
                       help='Detection model type to use (yolo or faster_rcnn)')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size (n, s, m, l, x) - only used when model-type is yolo')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for training and inference')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='Number of GPUs to use for training')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for Faster R-CNN training')
    
    parser.add_argument('--export-format', type=str, default='onnx', choices=['onnx', 'engine'],
                        help='Format to export the model to (onnx, engine for TensorRT)')
    
    parser.add_argument('--model-path', type=str, nargs='+', 
                       help='Path(s) to pre-trained model(s) for testing or RTMP processing. For ensemble testing, provide multiple paths.')
    parser.add_argument('--sample-dir', type=str,
                       help='Directory with sample images for testing')
    parser.add_argument('--output-video', type=str,
                       help='Path to save output video')
    
    parser.add_argument('--follow-id', type=int, default=None,
                       help='Track ID of the object to generate follow commands for (single model RTMP mode only)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.project_dir, exist_ok=True)
    
    clear_gpu_memory()
    
    if args.mode == 'prepare':
        data_manager = DataManager(base_path=args.base_path)
        data_manager.download_visdrone()
        dataset_yaml = data_manager.convert_visdrone_to_yolo()
        print(f"Dataset preparation complete. YAML path: {dataset_yaml}")
        
    elif args.mode == 'train':
        data_manager = DataManager(base_path=args.base_path)
        
        yolo_dir = os.path.join(args.base_path, "visdrone_yolo")
        dataset_yaml = os.path.join(yolo_dir, "dataset.yaml")
        if not os.path.exists(dataset_yaml):
            print("Dataset not found. Preparing it first...")
            data_manager.download_visdrone()
            dataset_yaml = data_manager.convert_visdrone_to_yolo()
        
        if args.model_type == 'yolo':
            # Create project directory based on model type
            yolo_project_dir = os.path.join(args.project_dir, "yolo")
            os.makedirs(yolo_project_dir, exist_ok=True)
            
            trainer = ModelTrainer(project_dir=yolo_project_dir)
            result = trainer.train_yolo(
                dataset_yaml=dataset_yaml,
                model_size=args.model_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                img_size=args.img_size,
                num_gpus=args.num_gpus
            )
            
            visualizer = Visualizer()
            results_csv = os.path.join(result['run_dir'], 'results.csv')
            if os.path.exists(results_csv):
                visualizer.visualize_training_results(results_csv)
            
            val_dir = os.path.join(args.base_path, "visdrone", "VisDrone2019-DET-val", "images")
            if os.path.exists(val_dir):
                visualizer.test_model_on_samples(
                    model_path=result['best_model_path'],
                    sample_dir=val_dir,
                    num_samples=5,
                    conf_threshold=args.conf_threshold
                )
            
            models_dir = os.path.join(yolo_project_dir, "models")
            trainer.save_model(
                model_path=result['best_model_path'],
                save_dir=models_dir,
                new_name=f"visdrone_yolo_{args.model_size}.pt"
            )
            
            print(f"Training completed. Best YOLO model saved to {models_dir}")
            
        elif args.model_type == 'faster_rcnn':
            # Create project directory based on model type
            faster_rcnn_project_dir = os.path.join(args.project_dir, "faster_rcnn")
            os.makedirs(faster_rcnn_project_dir, exist_ok=True)
            
            trainer = FasterRCNNTrainer(project_dir=faster_rcnn_project_dir)
            result = trainer.train_faster_rcnn(
                dataset_yaml=dataset_yaml,
                epochs=args.epochs,
                batch_size=args.batch_size,
                img_size=args.img_size,
                num_gpus=args.num_gpus,
                learning_rate=args.learning_rate
            )
            
            visualizer = Visualizer()
            results_csv = os.path.join(result['run_dir'], 'results.csv')
            if os.path.exists(results_csv):
                visualizer.visualize_training_results(results_csv)
            
            models_dir = os.path.join(faster_rcnn_project_dir, "models")
            trainer.save_model(
                model_path=result['best_model_path'],
                save_dir=models_dir,
                new_name=f"visdrone_faster_rcnn.pt"
            )
            
            print(f"Training completed. Best Faster R-CNN model saved to {models_dir}")
        
    elif args.mode == 'test':
        if not args.model_path:
            print("Error: --model-path is required for test mode")
            return
            
        sample_dir = args.sample_dir or os.path.join(args.base_path, "visdrone", "VisDrone2019-DET-val", "images")
        if not os.path.exists(sample_dir):
            print(f"Error: Sample directory {sample_dir} not found")
            return
            
        visualizer = Visualizer()
        visualizer.test_model_on_samples(
            model_path=args.model_path,
            sample_dir=sample_dir,
            num_samples=5,
            conf_threshold=args.conf_threshold
        )
        
    elif args.mode == 'rtmp':
        if not args.model_path:
            print("Error: --model-path is required for rtmp mode")
            return
            
        if args.follow_id is not None and len(args.model_path) > 1:
            print("Error: --follow-id can only be used with a single model path (ensemble mode not supported for following).")
            return
            
        # Detect if using Faster R-CNN model
        is_faster_rcnn = any("faster_rcnn" in path for path in args.model_path)
        
        # Prevent using --follow-id with Faster R-CNN models
        if is_faster_rcnn and args.follow_id is not None:
            print("Error: --follow-id is not supported with Faster R-CNN models.")
            return
            
        if is_faster_rcnn and len(args.model_path) == 1:
            # Process with a single Faster R-CNN model
            processor = RTMPProcessor(model_path=args.model_path[0], model_type='faster_rcnn')
        else:
            # Process with YOLO or ensemble
            processor = RTMPProcessor(model_path=args.model_path[0])
        
        processor.install_ffmpeg()
        
        results = processor.process_feed(
            rtmp_url=args.rtmp_url,
            output_path=args.output_video,
            confidence_threshold=args.conf_threshold,
            follow_target_id=args.follow_id 
        )
        
        if results:
            processor.display_results(results)
            
    elif args.mode == 'export':
        if not args.model_path:
            print("Error: --model-path pointing to a .pt file is required for export mode")
            return
            
        model_path = args.model_path[0]  # Get the first model path from the list
        if not model_path.endswith('.pt'):
            print("Error: Model path must point to a .pt file for export mode")
            return
            
        # Check if model is Faster R-CNN
        if "faster_rcnn" in model_path:
            print("Export not currently supported for Faster R-CNN models")
            return
            
        print(f"Exporting model {model_path} to {args.export_format} format...")
        try:
            model = YOLO(model_path) 
            export_path = model.export(format=args.export_format)
            print(f"Model exported successfully to: {export_path}")
        except Exception as e:
            print(f"Error during model export: {e}")
            print("Ensure the required backend libraries (e.g., onnx, onnxruntime, tensorrt) are installed.")
        
if __name__ == "__main__":
    main() 