import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from faster_rcnn import FasterRCNNWrapper

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
        
        # Initialize models list and is_ensemble flag
        self.models = []
        self.is_ensemble = False
        
        try:
            # If model_path is a list, we'll use ensemble mode
            if isinstance(model_path, list):
                self.is_ensemble = True
                for path in model_path:
                    if 'faster_rcnn' in path.lower():
                        model = FasterRCNNWrapper(path, class_names=self.class_names)
                    else:
                        model = YOLO(path)
                    self.models.append(model)
                print(f"Ensemble of {len(self.models)} models loaded")
            else:
                # Single model mode
                if self.model_type == 'faster_rcnn':
                    self.model = FasterRCNNWrapper(model_path, class_names=self.class_names)
                elif self.model_type == 'auto' and 'faster_rcnn' in model_path.lower():
                    self.model = FasterRCNNWrapper(model_path, class_names=self.class_names)
                else:
                    self.model = YOLO(model_path)
                self.models = [self.model]  # Add to models list for consistent access
                print(f"Model loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.models = []
            
    def install_ffmpeg(self):
        try:
            print("Installing FFmpeg...")
            os.system("apt-get update -qq && apt-get install -y ffmpeg")
            print("FFmpeg installed successfully")
            return True
        except Exception as e:
            print(f"Error installing FFmpeg: {e}")
            return False
            
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
                # Connect directly to the RTMP stream for real-time processing
                cap = cv2.VideoCapture(rtmp_url)
                
                # Set buffer size to minimize latency for real-time display
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                # Original behavior - download a segment first to test
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
                # For real-time display, check if we're falling behind
                if realtime_display and len(process_times) > 0:
                    avg_process_time = sum(process_times) / len(process_times)
                    frame_interval = 1.0 / fps if fps > 0 else 0.033  # Default to ~30fps if unknown
                    
                    # If processing is taking longer than frame interval, we need to skip frames
                    if avg_process_time > frame_interval:
                        skip_count = min(int(avg_process_time / frame_interval), max_frame_skip)
                        # Skip frames to catch up to real-time
                        for _ in range(skip_count):
                            ret, _ = cap.read()
                            if not ret:
                                break
                            frame_count += 1
                            skipped_frames += 1
                
                # Read the next frame
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
                        results = model.predict(frame, conf=confidence_threshold, verbose=False)
                        # Check if results is not empty and has valid boxes
                        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                            all_pred_boxes.extend(results[0].boxes.xyxy.cpu().tolist())
                            all_pred_scores.extend(results[0].boxes.conf.cpu().tolist())
                            all_pred_labels.extend(results[0].boxes.cls.cpu().tolist())
                    
                    final_boxes, final_scores, final_labels = self._apply_ensemble_nms(
                        all_pred_boxes, all_pred_scores, all_pred_labels, nms_iou_threshold
                    )
                else: 
                    model = self.models[0]
                    
                    # Handle differently if Faster R-CNN or YOLO
                    if isinstance(model, FasterRCNNWrapper):
                        # FasterRCNNWrapper doesn't support tracking
                        results = model.predict(frame, conf=confidence_threshold, verbose=False)
                        
                        # Check if results is not empty and has valid boxes
                        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                            boxes = results[0].boxes.xyxy.cpu().tolist()
                            confs = results[0].boxes.conf.cpu().tolist()
                            clss = results[0].boxes.cls.cpu().tolist()
                            final_boxes, final_scores, final_labels = boxes, confs, clss
                    else:
                        # YOLO with tracking
                        results = model.track(frame, persist=True, conf=confidence_threshold, verbose=False)
                        
                        # Check if results is not empty and has valid boxes with IDs
                        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None: 
                                boxes_data = results[0].boxes.data.cpu() 
                                for row in boxes_data:
                                    box = row[:4].tolist()
                                    track_id = int(row[4])
                                    conf = float(row[5])
                                    cls_id = int(row[6])
                                    
                                    final_boxes.append(box)
                                    final_scores.append(conf)
                                    final_labels.append(cls_id)
                                    final_track_ids.append(track_id)
                                    unique_track_ids.add(track_id)
                                    
                                    if follow_target_id is not None and track_id == follow_target_id:
                                        target_box_coords = box 
                            elif hasattr(results[0].boxes, 'xyxy'):
                                boxes = results[0].boxes.xyxy.cpu().tolist()
                                confs = results[0].boxes.conf.cpu().tolist()
                                clss = results[0].boxes.cls.cpu().tolist()
                                final_boxes, final_scores, final_labels = boxes, confs, clss
                
                annotated_frame = frame.copy()
                current_frame_detection_counts = {class_name: 0 for class_name in self.class_names} 

                for i, box in enumerate(final_boxes):
                    x1, y1, x2, y2 = map(int, box)
                    score = final_scores[i]
                    label_idx = int(final_labels[i])
                    cls_name = self.class_names[label_idx] if 0 <= label_idx < len(self.class_names) else f"class_{label_idx}"
                    color = self.colors[label_idx % len(self.colors)]

                    if cls_name in current_frame_detection_counts:
                        current_frame_detection_counts[cls_name] += 1
                    if cls_name in total_detection_counts:
                        total_detection_counts[cls_name] += 1

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
                    
                    label = f"{cls_name}: {score:.2f}"
                    # Only show track IDs for YOLO model in single model mode
                    if not self.is_ensemble and i < len(final_track_ids) and not isinstance(self.models[0], FasterRCNNWrapper):
                        label = f"ID:{final_track_ids[i]} {label}"
                    
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                drone_command = "N/A" 
                # Following only works with single YOLO model
                if follow_target_id is not None and not self.is_ensemble and not isinstance(self.models[0], FasterRCNNWrapper):
                    drone_command = self._get_control_commands(
                        frame_center_x, frame_center_y, target_box_coords 
                    )
                    cmd_text_pos = (int(frame_center_x) - 150, frame_height - 20) 
                    (w, h), _ = cv2.getTextSize(f"CMD: {drone_command}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated_frame, (cmd_text_pos[0]-10, cmd_text_pos[1]-h-5), (cmd_text_pos[0]+w+10, cmd_text_pos[1]+10), (0,0,0), -1)
                    cv2.putText(annotated_frame, f"CMD: {drone_command}", cmd_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) 
                
                elapsed_time = time.time() - start_time
                fps_actual = processed_count / elapsed_time if elapsed_time > 0 else 0
                cv2.rectangle(annotated_frame, (10, 10), (250, 40), (0, 0, 0), -1)
                cv2.putText(annotated_frame, f"FPS: {fps_actual:.1f} ({skipped_frames} skipped)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                y_offset = 50
                if self.is_ensemble:
                    ensemble_info = "Ensemble Mode"
                    cv2.rectangle(annotated_frame, (10, y_offset), (180, y_offset+30), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, ensemble_info, (15, y_offset+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    y_offset += 40
                elif isinstance(self.models[0], FasterRCNNWrapper):
                    model_info = "Faster R-CNN"
                    cv2.rectangle(annotated_frame, (10, y_offset), (180, y_offset+30), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, model_info, (15, y_offset+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    y_offset += 40
                
                # Display detections count on frame
                for cls_name, count in {k: current_frame_detection_counts[k] for k in current_frame_detection_counts if current_frame_detection_counts[k] > 0}.items():
                    cv2.rectangle(annotated_frame, (10, y_offset), (180, y_offset+30), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, f"{cls_name}: {count}", (15, y_offset+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 40
                
                # Calculate and track processing time
                process_end = time.time()
                process_time = process_end - process_start
                process_times.append(process_time)
                if len(process_times) > 30:  # Keep a rolling average of recent frames
                    process_times.pop(0)
                
                processed_count += 1
                
                # Display real-time window
                if realtime_display:
                    cv2.imshow('Real-time Drone Detection', annotated_frame)
                    # Press 'q' to exit the real-time display
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                # Save frames for display at the end if not in real-time mode
                elif frame_count % 10 == 0 or frame_count < 5: 
                    frames_to_display.append(annotated_frame.copy())
                
                if out:
                    out.write(annotated_frame)
                
                if frame_count % 10 == 0:
                    avg_process = sum(process_times) / len(process_times) if process_times else 0
                    print(f"Processed {processed_count}/{frame_count} frames (skipped {skipped_frames}). FPS: {fps_actual:.1f}, Avg process time: {avg_process:.3f}s")
                    
            # Release resources
            cap.release()
            
            if out:
                out.release()
            
            if realtime_display:
                cv2.destroyAllWindows()
                
            elapsed_time = time.time() - start_time
            results = {
                "frame_count": frame_count,
                "processed_count": processed_count,
                "skipped_frames": skipped_frames,
                "elapsed_time": elapsed_time,
                "fps": processed_count / elapsed_time if elapsed_time > 0 else 0,
                "total_detection_counts": total_detection_counts, 
                "sorted_total_detections": sorted([(k,v) for k,v in total_detection_counts.items() if v > 0], key=lambda item: item[1], reverse=True),
                "frames_to_display": frames_to_display if not realtime_display else [],
                "total_unique_tracks": len(unique_track_ids) if not self.is_ensemble and not isinstance(self.models[0], FasterRCNNWrapper) else None
            }
            return results

        except Exception as e:
            print(f"Error processing RTMP stream: {e}")
            return None

        finally:
            if not realtime_display and 'temp_file' in locals() and os.path.exists(temp_file):
                os.remove(temp_file)
                
    def _apply_ensemble_nms(self, boxes, scores, labels, iou_threshold=0.5):
        """
        Apply non-maximum suppression to ensemble predictions
        """
        if not boxes:
            return [], [], []
            
        # Convert to numpy arrays for processing
        boxes = np.array(boxes)
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Apply NMS per class
        final_boxes = []
        final_scores = []
        final_labels = []
        
        for class_id in np.unique(labels):
            # Get detections for this class
            mask = labels == class_id
            class_boxes = boxes[mask]
            class_scores = scores[mask]
            class_labels = labels[mask]
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                class_boxes.tolist(), 
                class_scores.tolist(), 
                score_threshold=0.1,  # Low threshold as we already filtered
                nms_threshold=iou_threshold
            )
            
            # Convert from OpenCV 4.x format if needed
            if isinstance(indices, tuple) and len(indices) == 1:
                indices = indices[0]
            
            # Add to final results
            if len(indices) > 0:
                for idx in indices:
                    # Handle different return types from OpenCV versions
                    if isinstance(idx, np.ndarray):
                        idx = idx.item()
                    elif isinstance(idx, (list, tuple)) and len(idx) > 0:
                        idx = idx[0]
                        
                    final_boxes.append(class_boxes[idx].tolist())
                    final_scores.append(class_scores[idx].item())
                    final_labels.append(class_labels[idx].item())
        
        return final_boxes, final_scores, final_labels
                
    def display_results(self, results):
        if not results:
            print("No results to display.")
            return
            
        frames = results["frames_to_display"]
        num_frames = len(frames)
        rows = (num_frames + 2) // 3  

        plt.figure(figsize=(18, 6*rows))
        for i, frame in enumerate(frames):
            plt.subplot(rows, 3, i+1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f"Frame {i*10 if i > 5 else i}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        print(f"\nProcessing completed. Total frames: {results['frame_count']}")
        print(f"Average FPS: {results['fps']:.2f}")
        if 'total_unique_tracks' in results and results['total_unique_tracks'] is not None:
            print(f"Total Unique Objects Tracked: {results['total_unique_tracks']}")
            
    def _get_control_commands(self, frame_center_x, frame_center_y, target_box, target_area_ratio_threshold=0.05, dead_zone_ratio=0.1):
        if target_box is None:
            return "Target Lost"

        x1, y1, x2, y2 = target_box
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        target_width = x2 - x1
        target_height = y2 - y1
        target_area = target_width * target_height
        frame_area = frame_center_x * 2 * frame_center_y * 2 

        dx = target_center_x - frame_center_x
        dy = target_center_y - frame_center_y

        dead_zone_x = frame_center_x * 2 * dead_zone_ratio / 2
        dead_zone_y = frame_center_y * 2 * dead_zone_ratio / 2

        commands = []

        if dx > dead_zone_x:
            commands.append("Move Right")
        elif dx < -dead_zone_x:
            commands.append("Move Left")

        if dy > dead_zone_y: 
            commands.append("Move Down")
        elif dy < -dead_zone_y: 
            commands.append("Move Up")

        target_area_ratio = target_area / frame_area
        if target_area_ratio < (target_area_ratio_threshold * 0.8): 
            commands.append("Move Forward")
        elif target_area_ratio > (target_area_ratio_threshold * 1.2): 
            commands.append("Move Backward")
            

        if not commands:
            return "Hold Position"
        
        return ", ".join(commands) 