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

def clear_gpu_memory():
    """Clear GPU memory to avoid out-of-memory errors"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Drone Detection System')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['prepare', 'train', 'test', 'rtmp', 'export'],
                       default='train', help='Operation mode')
    
    # Data paths
    parser.add_argument('--base-path', type=str, default='.',
                       help='Base path for dataset storage')
    parser.add_argument('--project-dir', type=str, default='./visdrone_yolo_training',
                       help='Directory to save training results')
    parser.add_argument('--rtmp-url', type=str, default='rtmp://192.168.1.1/live/stream',
                       help='RTMP URL for live processing')
    
    # Model parameters
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size (n, s, m, l, x)')
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
    
    # Export options (New)
    parser.add_argument('--export-format', type=str, default='onnx', choices=['onnx', 'engine'],
                        help='Format to export the model to (onnx, engine for TensorRT)')
    
    # File paths
    parser.add_argument('--model-path', type=str, 
                       help='Path to pre-trained model for testing or RTMP processing')
    parser.add_argument('--sample-dir', type=str,
                       help='Directory with sample images for testing')
    parser.add_argument('--output-video', type=str,
                       help='Path to save output video')
    
    # Follow target (New)
    parser.add_argument('--follow-id', type=int, default=None,
                       help='Track ID of the object to generate follow commands for (single model RTMP mode only)')
    
    return parser.parse_args()

def main():
    """Main function to run the drone detection system"""
    args = parse_args()
    
    # Prepare environment
    os.makedirs(args.project_dir, exist_ok=True)
    
    # Clear GPU memory
    clear_gpu_memory()
    
    if args.mode == 'prepare':
        # Initialize data manager and prepare dataset
        data_manager = DataManager(base_path=args.base_path)
        data_manager.download_visdrone()
        dataset_yaml = data_manager.convert_visdrone_to_yolo()
        print(f"Dataset preparation complete. YAML path: {dataset_yaml}")
        
    elif args.mode == 'train':
        # Initialize data manager and prepare dataset
        data_manager = DataManager(base_path=args.base_path)
        
        # Check if dataset exists, if not, prepare it
        yolo_dir = os.path.join(args.base_path, "visdrone_yolo")
        dataset_yaml = os.path.join(yolo_dir, "dataset.yaml")
        if not os.path.exists(dataset_yaml):
            print("Dataset not found. Preparing it first...")
            data_manager.download_visdrone()
            dataset_yaml = data_manager.convert_visdrone_to_yolo()
        
        # Initialize model trainer and train
        trainer = ModelTrainer(project_dir=args.project_dir)
        result = trainer.train_yolo(
            dataset_yaml=dataset_yaml,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_gpus=args.num_gpus
        )
        
        # Display training results
        visualizer = Visualizer()
        results_csv = os.path.join(result['run_dir'], 'results.csv')
        if os.path.exists(results_csv):
            visualizer.visualize_training_results(results_csv)
        
        # Test on some validation samples
        val_dir = os.path.join(args.base_path, "visdrone", "VisDrone2019-DET-val", "images")
        if os.path.exists(val_dir):
            visualizer.test_model_on_samples(
                model_path=result['best_model_path'],
                sample_dir=val_dir,
                num_samples=5,
                conf_threshold=args.conf_threshold
            )
        
        # Save final model
        models_dir = os.path.join(args.project_dir, "models")
        trainer.save_model(
            model_path=result['best_model_path'],
            save_dir=models_dir,
            new_name=f"visdrone_yolo_{args.model_size}.pt"
        )
        
        print(f"Training completed. Best model saved to {models_dir}")
        
    elif args.mode == 'test':
        # Test model on sample images
        if not args.model_path:
            print("Error: --model-path is required for test mode")
            return
            
        # Construct sample_dir relative to base_path if not provided
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
        # Process RTMP feed
        if not args.model_path:
            print("Error: --model-path is required for rtmp mode")
            return
            
        # Check if ensemble mode is used with follow-id
        if args.follow_id is not None and len(args.model_path) > 1:
            print("Error: --follow-id can only be used with a single model path (ensemble mode not supported for following).")
            return
            
        # Initialize RTMP processor - uses first model if list provided (checked above)
        processor = RTMPProcessor(model_path=args.model_path[0])
        
        # Install FFmpeg if needed
        processor.install_ffmpeg()
        
        # Process the feed, passing the follow_id
        results = processor.process_feed(
            rtmp_url=args.rtmp_url,
            output_path=args.output_video,
            confidence_threshold=args.conf_threshold,
            follow_target_id=args.follow_id # Pass the ID
        )
        
        # Display results if available
        if results:
            processor.display_results(results)
            
        # Generate client script for local usage
        processor.generate_client_script()
        
    elif args.mode == 'export':
        # Export the model
        if not args.model_path or not args.model_path.endswith('.pt'):
            print("Error: --model-path pointing to a .pt file is required for export mode")
            return
            
        print(f"Exporting model {args.model_path} to {args.export_format} format...")
        try:
            model = YOLO(args.model_path) 
            export_path = model.export(format=args.export_format)
            print(f"Model exported successfully to: {export_path}")
        except Exception as e:
            print(f"Error during model export: {e}")
            print("Ensure the required backend libraries (e.g., onnx, onnxruntime, tensorrt) are installed.")
        
if __name__ == "__main__":
    main() 