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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description='Drone Detection System')
    
    parser.add_argument('--mode', 
                       type=str, 
                       choices=['prepare', 'train', 'test', 'rtmp', 'export'],
                       default='train')
    
    parser.add_argument('--base-path', 
                       type=str, 
                       default='.')
    
    parser.add_argument('--project-dir', 
                       type=str, 
                       default='./visdrone_yolo_training')
    
    parser.add_argument('--rtmp-url', 
                       type=str, 
                       default='rtmp://192.168.1.1/live/stream')
    
    parser.add_argument('--model-type', 
                       type=str, 
                       default='yolo', 
                       choices=['yolo'])
    
    parser.add_argument('--model-size', 
                       type=str, 
                       default='n', 
                       choices=['n', 's', 'm', 'l', 'x'])
    
    parser.add_argument('--epochs', 
                       type=int, 
                       default=100)
    
    parser.add_argument('--batch-size', 
                       type=int, 
                       default=16)
    
    parser.add_argument('--img-size', 
                       type=int, 
                       default=640)
    
    parser.add_argument('--num-gpus', 
                       type=int, 
                       default=1)
    
    parser.add_argument('--conf-threshold', 
                       type=float, 
                       default=0.25)
    
    parser.add_argument('--export-format', 
                       type=str, 
                       default='onnx', 
                       choices=['onnx'])
    
    parser.add_argument('--model-path', 
                       type=str, 
                       nargs='+')
    
    parser.add_argument('--sample-dir', 
                       type=str)
    
    parser.add_argument('--output-video', 
                       type=str)
    
    parser.add_argument('--follow-id', 
                       type=int, 
                       default=None)
    
    parser.add_argument('--realtime-display', 
                       action='store_true')
    
    parser.add_argument('--max-frame-skip', 
                       type=int, 
                       default=5)
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.project_dir, exist_ok=True)
    clear_gpu_memory()
    
    if args.mode == 'prepare':
        data_manager = DataManager(base_path=args.base_path)
        data_manager.download_visdrone()
        data_manager.convert_visdrone_to_yolo()
        
    elif args.mode == 'train':
        data_manager = DataManager(base_path=args.base_path)
        
        yolo_dir = os.path.join(args.base_path, "visdrone_yolo")
        dataset_yaml = os.path.join(yolo_dir, "dataset.yaml")
        
        if not os.path.exists(dataset_yaml):
            data_manager.download_visdrone()
            dataset_yaml = data_manager.convert_visdrone_to_yolo()
        
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
        
    elif args.mode == 'test':
        if not args.model_path:
            return
            
        if args.sample_dir:
            sample_dir = args.sample_dir
        else:
            sample_dir = os.path.join(args.base_path, "visdrone", "VisDrone2019-DET-val", "images")
            
        if not os.path.exists(sample_dir):
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
            return
            
        if len(args.model_path) == 1:
            model_path = args.model_path[0]
        else:
            model_path = args.model_path
            
        processor = RTMPProcessor(model_path=model_path)
                
        results = processor.process_feed(
            rtmp_url=args.rtmp_url,
            output_path=args.output_video,
            confidence_threshold=args.conf_threshold,
            follow_target_id=args.follow_id,
            realtime_display=args.realtime_display,
            max_frame_skip=args.max_frame_skip
        )
        
        if results and not args.realtime_display:
            processor.display_results(results)
            
    elif args.mode == 'export':
        if not args.model_path:
            return
            
        if not args.model_path[0].endswith('.pt'):
            return
            
        try:
            model = YOLO(args.model_path[0]) 
            export_path = model.export(format=args.export_format)
        except Exception as e:
            return
        
if __name__ == "__main__":
    main() 