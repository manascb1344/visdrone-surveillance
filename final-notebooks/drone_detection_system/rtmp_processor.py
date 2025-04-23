import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

class RTMPProcessor:
    
    def __init__(self, model_path, class_names=None, model_type='yolo'):
        self.model_path = model_path
        self.model_type = model_type
        self.class_names = class_names or [
            'person', 'bicycle', 'car', 'van', 
            'truck', 'tricycle', 'bus', 'motorcycle'
        ]
        
        self.colors = [
            (0, 255, 0),    
            (255, 0, 0),    
            (0, 0, 255),    
            (255, 255, 0),  
            (255, 0, 255),  
            (0, 255, 255),  
            (255, 165, 0),  
            (128, 0, 128)   
        ]
        
        self.models = []
        self.is_ensemble = False
        
        try:
            if isinstance(model_path, list):
                self.is_ensemble = True
                for path in model_path:
                    model = YOLO(path)
                    self.models.append(model)
                print(f"Ensemble of {len(self.models)} models loaded")
            else:
                self.model = YOLO(model_path)
                self.models = [self.model]
                print(f"Model loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.models = []
                        
    def process_feed(self, rtmp_url, output_path=None, confidence_threshold=0.25, max_frames=300, nms_iou_threshold=0.5, follow_target_id=None, realtime_display=False, max_frame_skip=5):
        if self.model is None:
            print("Error: Model not loaded.")
            return None
            
        print(f"Attempting to connect to RTMP feed: {rtmp_url}")
        
        if follow_target_id is not None and self.is_ensemble:
             print("Warning: Target following (--follow-id) is only supported in single-model mode. Disabling follow.")
             follow_target_id = None 
             
        if not self.models:
            print("Error: No models available for processing.")
            return None
            
        try:
            if realtime_display:
                print("Starting real-time RTMP stream processing...")
                cap = cv2.VideoCapture(rtmp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                temp_file = "/tmp/rtmp_temp.mp4"
                print("Downloading a segment from the RTMP stream for testing...")
                os.system(f"ffmpeg -hide_banner -loglevel error -y -i {rtmp_url} -t 10 -c copy {temp_file}")

                if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                    print("Failed to download RTMP stream. Check your URL and network access.")
                    return None

                print(f"Successfully downloaded segment to {temp_file}")
                cap = cv2.VideoCapture(temp_file)

            if not cap.isOpened():
                print("Error: Could not open video stream.")
                return None

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"Video properties: {frame_width}x{frame_height} at {fps} FPS")

            out = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            frame_count = 0
            processed_count = 0
            skipped_frames = 0
            total_detection_counts = {class_name: 0 for class_name in self.class_names} 
            unique_track_ids = set() 
            start_time = time.time()
            frames_to_display = []

            frame_center_x = frame_width / 2
            frame_center_y = frame_height / 2
            
            last_process_time = time.time()
            process_times = []

            while cap.isOpened() and (frame_count < max_frames or realtime_display):
                if realtime_display and len(process_times) > 0:
                    avg_process_time = sum(process_times) / len(process_times)
                    frame_interval = 1.0 / fps if fps > 0 else 0.033
                    
                    if avg_process_time > frame_interval:
                        skip_count = min(int(avg_process_time / frame_interval), max_frame_skip)
                        for _ in range(skip_count):
                            ret, _ = cap.read()
                            if not ret:
                                break
                            frame_count += 1
                            skipped_frames += 1
                
                ret, frame = cap.read()
                if not ret: 
                    break
                
                frame_count += 1
                process_start = time.time()

                target_box_coords = None 
                final_boxes = []
                final_scores = []
                final_labels = []
                final_track_ids = [] 
                
                if self.is_ensemble:
                    all_pred_boxes = []
                    all_pred_scores = []
                    all_pred_labels = []
                    for model in self.models:
                        results = model.predict(frame, conf=confidence_threshold)
                        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                            print(f"Ensemble model detected {len(results[0].boxes.xyxy)} boxes")
                            all_pred_boxes.extend(results[0].boxes.xyxy.cpu().tolist())
                            all_pred_scores.extend(results[0].boxes.conf.cpu().tolist())
                            all_pred_labels.extend(results[0].boxes.cls.cpu().tolist())
                    
                    final_boxes, final_scores, final_labels = self._apply_ensemble_nms(
                        all_pred_boxes, all_pred_scores, all_pred_labels, nms_iou_threshold
                    )
                    print(f"After NMS: {len(final_boxes)} boxes")
                else: 
                    model = self.models[0]
                    
                    if follow_target_id is not None:
                        results = model.track(frame, persist=True, conf=confidence_threshold)
                    else:
                        results = model.predict(frame, conf=confidence_threshold)
                    
                    if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        detected_boxes = results[0].boxes
                        
                        if hasattr(detected_boxes, 'id') and detected_boxes.id is not None and follow_target_id is not None:
                            tracking_ids = detected_boxes.id.cpu().numpy().astype(int)
                            target_box_idx = np.where(tracking_ids == follow_target_id)[0]
                            
                            if len(target_box_idx) > 0:
                                idx = target_box_idx[0]
                                if idx < len(detected_boxes.xyxy):
                                    target_box = detected_boxes.xyxy[idx].cpu().numpy().astype(int)
                                    target_box_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
                                    target_box_area_ratio = target_box_area / (frame_width * frame_height)
                                    target_box_coords = target_box
                        
                        final_boxes = detected_boxes.xyxy.cpu().tolist()
                        final_scores = detected_boxes.conf.cpu().tolist()
                        final_labels = detected_boxes.cls.cpu().tolist()
                        
                        if hasattr(detected_boxes, 'id') and detected_boxes.id is not None:
                            final_track_ids = detected_boxes.id.cpu().tolist()
                
                draw_frame = frame.copy()
                
                if len(final_boxes) > 0:
                    for i, (box, score, label_idx) in enumerate(zip(final_boxes, final_scores, final_labels)):
                        label_idx = int(label_idx)
                        if label_idx < len(self.class_names):
                            label = self.class_names[label_idx]
                            total_detection_counts[label] += 1
                        else:
                            label = f"Class {label_idx}"
                            
                        color = self.colors[label_idx % len(self.colors)]
                        x1, y1, x2, y2 = map(int, box)
                        
                        track_id = None
                        if i < len(final_track_ids):
                            track_id = int(final_track_ids[i])
                            unique_track_ids.add(track_id)
                            
                        text = f"{label} {score:.2f}" + (f" ID:{track_id}" if track_id is not None else "")
                        
                        cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.rectangle(draw_frame, (x1, y1-20), (x1+len(text)*8, y1), color, -1)
                        cv2.putText(draw_frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        
                        if target_box_coords is not None and final_track_ids and i < len(final_track_ids) and final_track_ids[i] == follow_target_id:
                            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            cv2.line(draw_frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
                            cv2.line(draw_frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
                            
                            target_center = (center_x, center_y)
                            frame_center = (frame_width // 2, frame_height // 2)
                            
                            cv2.line(draw_frame, frame_center, target_center, (0, 255, 255), 2)
                            
                            control_commands = self._get_control_commands(
                                frame_center_x, frame_center_y, box)
                            
                            for idx, cmd in enumerate(control_commands):
                                y_pos = 30 + idx * 30
                                cv2.putText(draw_frame, cmd, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.7, (0, 255, 255), 2)
                
                cv2.putText(draw_frame, f"Frame: {frame_count}", (10, frame_height - 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                process_end = time.time()
                process_time = process_end - process_start
                fps_text = f"Processing FPS: {1/process_time:.2f}" if process_time > 0 else "Processing FPS: N/A"
                cv2.putText(draw_frame, fps_text, (10, frame_height - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(draw_frame, f"Objects: {len(final_boxes)}", (10, frame_height - 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                           
                process_times.append(process_time)
                if len(process_times) > 10:
                    process_times.pop(0)
                
                if out:
                    out.write(draw_frame)
                    
                if realtime_display:
                    cv2.imshow('RTMP Stream', draw_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                else:
                    frames_to_display.append(draw_frame)
                    processed_count += 1

                last_process_time = time.time()
            
            cap.release()
            if out:
                out.write(draw_frame)
                out.release()
                
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            
            results = {
                'frames': frames_to_display[:min(6, len(frames_to_display))],
                'processed_frames': processed_count,
                'skipped_frames': skipped_frames,
                'total_frames': frame_count,
                'processing_time': total_time,
                'detection_counts': total_detection_counts,
                'unique_tracks': len(unique_track_ids)
            }
            
            print(f"\nProcessed {processed_count} frames, skipped {skipped_frames} frames")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average FPS: {processed_count/total_time:.2f}" if total_time > 0 else "Average FPS: N/A")
            print("\nDetection counts by class:")
            for cls, count in total_detection_counts.items():
                if count > 0:
                    print(f"  {cls}: {count}")
                    
            if len(unique_track_ids) > 0:
                print(f"\nUnique tracked objects: {len(unique_track_ids)}")
            
            if output_path:
                print(f"\nOutput video saved to: {output_path}")
            
            return results
            
        except Exception as e:
            print(f"Error processing RTMP feed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _apply_ensemble_nms(self, boxes, scores, labels, iou_threshold=0.5):
        if not boxes:
            return [], [], []
            
        boxes = np.array(boxes)
        scores = np.array(scores)
        labels = np.array(labels)
        
        indices = np.argsort(-scores)
        boxes = boxes[indices]
        scores = scores[indices]
        labels = labels[indices]
        
        keep_boxes = []
        keep_scores = []
        keep_labels = []
        
        while boxes.shape[0] > 0:
            pick = 0
            current_box = boxes[pick]
            current_score = scores[pick]
            current_label = labels[pick]
            
            keep_boxes.append(current_box.tolist())
            keep_scores.append(current_score.item())
            keep_labels.append(current_label.item())
            
            remainder_indices = np.where(labels == current_label)[0][1:]
            
            if len(remainder_indices) == 0:
                boxes = np.delete(boxes, 0, axis=0)
                scores = np.delete(scores, 0)
                labels = np.delete(labels, 0)
                continue
                
            remainder_boxes = boxes[remainder_indices]
            
            xx1 = np.maximum(current_box[0], remainder_boxes[:, 0])
            yy1 = np.maximum(current_box[1], remainder_boxes[:, 1])
            xx2 = np.minimum(current_box[2], remainder_boxes[:, 2])
            yy2 = np.minimum(current_box[3], remainder_boxes[:, 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            area1 = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            area2 = (remainder_boxes[:, 2] - remainder_boxes[:, 0]) * (remainder_boxes[:, 3] - remainder_boxes[:, 1])
            union = area1 + area2 - intersection
            
            iou = intersection / union
            
            overlap_indices = np.where(iou > iou_threshold)[0]
            delete_indices = remainder_indices[overlap_indices]
            
            delete_indices = np.append([0], delete_indices)
            
            boxes = np.delete(boxes, delete_indices, axis=0)
            scores = np.delete(scores, delete_indices)
            labels = np.delete(labels, delete_indices)
            
        return keep_boxes, keep_scores, keep_labels
            
    def display_results(self, results):
        if not results or 'frames' not in results or not results['frames']:
            print("No frames to display.")
            return
            
        frames = results['frames']
        num_frames = len(frames)
        
        cols = min(3, num_frames)
        rows = (num_frames + cols - 1) // cols
        
        plt.figure(figsize=(15, 5 * rows))
        
        for i, frame in enumerate(frames):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f"Frame {i+1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        if 'detection_counts' in results:
            counts = results['detection_counts']
            fig, ax = plt.subplots(figsize=(10, 5))
            
            classes = [cls for cls, count in counts.items() if count > 0]
            counts = [count for cls, count in counts.items() if count > 0]
            
            if classes:
                ax.bar(classes, counts)
                ax.set_ylabel('Count')
                ax.set_xlabel('Class')
                ax.set_title('Detections by Class')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
                
    def _get_control_commands(self, frame_center_x, frame_center_y, target_box, target_area_ratio_threshold=0.05, dead_zone_ratio=0.1):
        commands = []
        
        target_center_x = (target_box[0] + target_box[2]) / 2
        target_center_y = (target_box[1] + target_box[3]) / 2
        
        dead_zone_width = frame_center_x * dead_zone_ratio * 2
        dead_zone_height = frame_center_y * dead_zone_ratio * 2
        
        if target_center_x < frame_center_x - dead_zone_width/2:
            commands.append("Move Left")
        elif target_center_x > frame_center_x + dead_zone_width/2:
            commands.append("Move Right")
            
        if target_center_y < frame_center_y - dead_zone_height/2:
            commands.append("Move Up")
        elif target_center_y > frame_center_y + dead_zone_height/2:
            commands.append("Move Down")
            
        target_width = target_box[2] - target_box[0]
        target_height = target_box[3] - target_box[1]
        target_area = target_width * target_height
        frame_area = frame_center_x * 2 * frame_center_y * 2
        target_area_ratio = target_area / frame_area
        
        if target_area_ratio < target_area_ratio_threshold:
            commands.append("Move Closer")
        elif target_area_ratio > target_area_ratio_threshold * 4:
            commands.append("Move Away")
            
        if not commands:
            commands.append("Hold Position")
            
        return commands 