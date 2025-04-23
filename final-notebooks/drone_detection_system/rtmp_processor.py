import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

class RTMPProcessor:
    def __init__(self, model_path, class_names=None, model_type='yolo'):
        self.model_path = model_path
        self.model_type = model_type
        self.follow_target_id = None
        self.available_ids = []
        self.show_help = False
        
        self.input_active = False
        self.input_text = ""
        self.input_rect = [0, 0, 0, 0]
        
        if class_names is None:
            self.class_names = [
                'person', 'bicycle', 'car', 'van', 
                'truck', 'tricycle', 'bus', 'motorcycle'
            ]
        else:
            self.class_names = class_names
        
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (255, 165, 0),  # Orange
            (128, 0, 128)   # Purple
        ]
        
        self.models = []
        self.is_ensemble = False
        
        try:
            if isinstance(model_path, list):
                self.is_ensemble = True
                for path in model_path:
                    model = YOLO(path)
                    self.models.append(model)
            else:
                self.model = YOLO(model_path)
                self.models = [self.model]
        except Exception as e:
            self.model = None
            self.models = []
                        
    def set_follow_target_id(self, target_id):
        if target_id in self.available_ids or target_id is None:
            self.follow_target_id = target_id
            
        return self.follow_target_id
                        
    def process_feed(self, rtmp_url, output_path=None, confidence_threshold=0.25, 
                    max_frames=300, nms_iou_threshold=0.5, follow_target_id=None, 
                    realtime_display=False, max_frame_skip=5):
        if not self.models:
            return None
            
        if follow_target_id is not None:
            self.follow_target_id = follow_target_id
            
        if self.follow_target_id is not None and self.is_ensemble:
            self.follow_target_id = None 
            
        try:
            cap, frame_width, frame_height, fps = self._open_video_stream(rtmp_url, realtime_display)
            if cap is None:
                return None

            out = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                if fps <= 0 or fps > 60: 
                    fps = 30.0
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                process_all_frames = True
            else:
                process_all_frames = False

            if realtime_display:
                cv2.namedWindow('RTMP Stream', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('RTMP Stream', 1280, 720)
                cv2.setMouseCallback('RTMP Stream', self._mouse_callback)

            frame_count = 0
            processed_count = 0
            skipped_frames = 0
            total_detection_counts = {}
            
            for class_name in self.class_names:
                total_detection_counts[class_name] = 0
                
            unique_track_ids = set()
            self.available_ids = []
            start_time = time.time()
            frames_to_display = []
            
            frame_center_x = frame_width / 2
            frame_center_y = frame_height / 2
            
            last_process_time = time.time()
            process_times = []

            while cap.isOpened() and (frame_count < max_frames or realtime_display or (process_all_frames and output_path)):
                if realtime_display and not output_path and len(process_times) > 0:
                    avg_process_time = sum(process_times) / len(process_times)
                    
                    if fps > 0:
                        frame_interval = 1.0 / fps
                    else:
                        frame_interval = 0.033
                    
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

                final_boxes, final_scores, final_labels, final_track_ids = self._process_frame(
                    frame, confidence_threshold, nms_iou_threshold, self.follow_target_id)
                
                self.available_ids = []
                for track_id in final_track_ids:
                    if track_id is not None:
                        track_id_int = int(track_id)
                        unique_track_ids.add(track_id_int)
                        if track_id_int not in self.available_ids:
                            self.available_ids.append(track_id_int)
                
                self.available_ids.sort()
                
                for label_idx in final_labels:
                    label_idx = int(label_idx)
                    if label_idx < len(self.class_names):
                        class_name = self.class_names[label_idx]
                        total_detection_counts[class_name] += 1
                
                draw_frame = self._draw_detections(
                    frame, 
                    final_boxes, 
                    final_scores, 
                    final_labels, 
                    final_track_ids, 
                    self.follow_target_id
                )
                
                process_end = time.time()
                process_time = process_end - process_start
                
                draw_frame = self._add_frame_info(
                    draw_frame, 
                    frame_count, 
                    process_time, 
                    len(final_boxes), 
                    frame_height,
                    final_labels,
                    frame_width
                )
                
                if realtime_display:
                    draw_frame = self._add_ui_controls(draw_frame, frame_width, frame_height)
                
                process_times.append(process_time)
                if len(process_times) > 10:
                    process_times.pop(0)
                
                if out:
                    out.write(draw_frame)
                    
                if realtime_display:
                    cv2.imshow('RTMP Stream', draw_frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if self.input_active:
                        self._handle_input_key(key)
                    else:
                        self._handle_key_press(key)
                elif not output_path:
                    frames_to_display.append(draw_frame)
                    processed_count += 1
                else:
                    processed_count += 1

                last_process_time = time.time()
            
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            
            display_frames = frames_to_display
            if len(display_frames) > 6:
                display_frames = display_frames[:6]
                
            results = {
                'frames': display_frames,
                'processed_frames': processed_count,
                'skipped_frames': skipped_frames,
                'total_frames': frame_count,
                'processing_time': total_time,
                'detection_counts': total_detection_counts,
                'unique_tracks': len(unique_track_ids)
            }
            
            return results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
    
    def _handle_input_key(self, key):
        if key == 13:
            self._apply_input_text()
            
        elif key == 27:
            self.input_active = False
            self.input_text = ""
            
        elif key == 8:
            if self.input_text:
                self.input_text = self.input_text[:-1]
            
        elif (48 <= key <= 57) or key == 45:
            self.input_text += chr(key)
    
    def _handle_key_press(self, key):
        if key == ord('i') or key == ord('I'):
            self.input_active = True
            self.input_text = ""
            
        elif key == ord('n') or key == ord('N'):
            if not self.available_ids:
                return
                
            if self.follow_target_id is None:
                self.follow_target_id = self.available_ids[0]
            else:
                try:
                    current_idx = self.available_ids.index(self.follow_target_id)
                    next_idx = (current_idx + 1) % len(self.available_ids)
                    self.follow_target_id = self.available_ids[next_idx]
                except ValueError:
                    self.follow_target_id = self.available_ids[0]
        
        elif key == ord('p') or key == ord('P'):
            if not self.available_ids:
                return
                
            if self.follow_target_id is None:
                self.follow_target_id = self.available_ids[-1]
            else:
                try:
                    current_idx = self.available_ids.index(self.follow_target_id)
                    prev_idx = (current_idx - 1) % len(self.available_ids)
                    self.follow_target_id = self.available_ids[prev_idx]
                except ValueError:
                    self.follow_target_id = self.available_ids[-1]
        
        elif key >= ord('0') and key <= ord('9'):
            selected = key - ord('0')
            if selected < len(self.available_ids):
                self.follow_target_id = self.available_ids[selected]
            
        elif key == ord('c') or key == ord('C'):
            self.follow_target_id = None
            
        elif key == ord('h') or key == ord('H'):
            self.show_help = not self.show_help
            
        elif key == ord('q') or key == 27:
            pass
        
    def _add_ui_controls(self, frame, frame_width, frame_height):
        ui_overlay = frame.copy()
        
        if self.show_help:
            ui_height = 130
        else:
            ui_height = 70
            
        cv2.rectangle(
            ui_overlay, 
            (0, frame_height - ui_height), 
            (int(frame_width), frame_height), 
            (0, 0, 0), 
            -1
        )
        
        frame = cv2.addWeighted(ui_overlay, 0.7, frame, 0.3, 0, frame)
        
        base_y_pos = frame_height - 45
        
        if self.follow_target_id is not None:
            target_text = f"Following ID: {self.follow_target_id}"
        else:
            target_text = "Following ID: None"
            
        cv2.putText(
            frame, 
            target_text, 
            (10, base_y_pos), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 255), 
            2
        )
        
        available_text = "Available IDs: "
        for i, id_val in enumerate(self.available_ids):
            if id_val == self.follow_target_id:
                available_text += f"[{i}:{id_val}] "
            else:
                available_text += f"{i}:{id_val} "
        
        cv2.putText(
            frame, 
            available_text, 
            (10, base_y_pos + 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 255), 
            1
        )
        
        input_field_x = frame_width - 200
        input_field_width = 180
        input_field_y = frame_height - 45
        input_field_height = 30
        
        self.input_rect = [input_field_x, input_field_y, input_field_width, input_field_height]
        
        if self.input_active:
            field_color = (0, 255, 255)
        else:
            field_color = (100, 100, 100)
            
        cv2.rectangle(
            frame, 
            (input_field_x, input_field_y), 
            (input_field_x + input_field_width, input_field_y + input_field_height), 
            field_color, 
            -1
        )
        
        cv2.rectangle(
            frame, 
            (input_field_x, input_field_y), 
            (input_field_x + input_field_width, input_field_y + input_field_height), 
            (200, 200, 200), 
            1
        )
        
        label_x = input_field_x - 80
        cv2.putText(
            frame, 
            "Target ID:", 
            (label_x, input_field_y + 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 255), 
            1
        )
        
        if self.input_active:
            text_color = (0, 0, 0)
        else:
            text_color = (0, 255, 255)
            
        if self.input_text:
            display_text = self.input_text
        elif not self.input_active:
            display_text = "Enter ID"
            text_color = (150, 150, 150)
        else:
            display_text = "|"
        
        cv2.putText(
            frame, 
            display_text, 
            (input_field_x + 10, input_field_y + 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            text_color, 
            1
        )
        
        if self.show_help:
            help_text = [
                "Controls: [N]ext ID, [P]revious ID, [0-9] Select ID by index",
                "[I] Input ID directly, [C]lear/stop following, [H] Toggle help, [Q] Quit"
            ]
            for i, text in enumerate(help_text):
                y_position = base_y_pos - 20 - (i * 20)
                cv2.putText(
                    frame, 
                    text, 
                    (10, y_position), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 255), 
                    1
                )
        
        return frame
    
    def _open_video_stream(self, rtmp_url, realtime_display):
        try:
            if realtime_display:
                cap = cv2.VideoCapture(rtmp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                temp_file = "/tmp/rtmp_temp.mp4"
                ffmpeg_command = f"ffmpeg -hide_banner -loglevel error -y -i {rtmp_url} -vsync 0 -c copy {temp_file}"
                os.system(ffmpeg_command)

                if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                    return None, None, None, None

                cap = cv2.VideoCapture(temp_file)

            if not cap.isOpened():
                return None, None, None, None

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            return cap, frame_width, frame_height, fps
            
        except Exception as e:
            return None, None, None, None
    
    def _process_frame(self, frame, confidence_threshold, nms_iou_threshold, follow_target_id):
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
                
                if results and len(results) > 0:
                    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        all_pred_boxes.extend(results[0].boxes.xyxy.cpu().tolist())
                        all_pred_scores.extend(results[0].boxes.conf.cpu().tolist())
                        all_pred_labels.extend(results[0].boxes.cls.cpu().tolist())
            
            final_boxes, final_scores, final_labels = self._apply_ensemble_nms(
                all_pred_boxes, 
                all_pred_scores, 
                all_pred_labels, 
                nms_iou_threshold
            )
        else:
            model = self.models[0]
            results = model.track(frame, persist=True, conf=confidence_threshold)
            
            if results and len(results) > 0:
                if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    detected_boxes = results[0].boxes
                    
                    if hasattr(detected_boxes, 'id') and detected_boxes.id is not None:
                        if follow_target_id is not None:
                            tracking_ids = detected_boxes.id.cpu().numpy().astype(int)
                            target_box_idx = np.where(tracking_ids == follow_target_id)[0]
                            
                            if len(target_box_idx) > 0:
                                idx = target_box_idx[0]
                                if idx < len(detected_boxes.xyxy):
                                    target_box = detected_boxes.xyxy[idx].cpu().numpy().astype(int)
                                    target_box_coords = target_box
                
                    final_boxes = detected_boxes.xyxy.cpu().tolist()
                    final_scores = detected_boxes.conf.cpu().tolist()
                    final_labels = detected_boxes.cls.cpu().tolist()
                    
                    if hasattr(detected_boxes, 'id') and detected_boxes.id is not None:
                        final_track_ids = detected_boxes.id.cpu().tolist()
        
        return final_boxes, final_scores, final_labels, final_track_ids
    
    def _draw_detections(self, frame, boxes, scores, labels, track_ids, follow_target_id=None):
        draw_frame = frame.copy()
        
        if not boxes:
            return draw_frame
            
        for i, (box, score, label_idx) in enumerate(zip(boxes, scores, labels)):
            label_idx = int(label_idx)
            
            if label_idx < len(self.class_names):
                label = self.class_names[label_idx]
            else:
                label = f"Class {label_idx}"
                
            color = self.colors[label_idx % len(self.colors)]
            x1, y1, x2, y2 = map(int, box)
            
            track_id = None
            if i < len(track_ids):
                track_id = int(track_ids[i])
            
            if track_id is not None:
                text = f"{label} {score:.2f} ID:{track_id}"
            else:
                text = f"{label} {score:.2f}"
            
            cv2.rectangle(
                draw_frame, 
                (x1, y1), 
                (x2, y2), 
                color, 
                1
            )
            
            cv2.rectangle(
                draw_frame, 
                (x1, y1-20), 
                (x1+len(text)*8, y1), 
                color, 
                -1
            )
            
            cv2.putText(
                draw_frame, 
                text, 
                (x1, y1-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2
            )
            
            if track_id is not None and track_id == follow_target_id:
                cv2.rectangle(
                    draw_frame, 
                    (x1, y1), 
                    (x2, y2), 
                    (0, 255, 255), 
                    4
                )
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                cv2.line(
                    draw_frame, 
                    (center_x - 20, center_y), 
                    (center_x + 20, center_y), 
                    (0, 255, 255), 
                    2
                )
                
                cv2.line(
                    draw_frame, 
                    (center_x, center_y - 20), 
                    (center_x, center_y + 20), 
                    (0, 255, 255), 
                    2
                )
                
                frame_center = (draw_frame.shape[1] // 2, draw_frame.shape[0] // 2)
                cv2.line(
                    draw_frame, 
                    frame_center, 
                    (center_x, center_y), 
                    (0, 255, 255), 
                    2
                )
                
                control_commands = self._get_control_commands(
                    frame_center[0], 
                    frame_center[1], 
                    box
                )
                
                for idx, cmd in enumerate(control_commands):
                    y_pos = 30 + idx * 30
                    cv2.putText(
                        draw_frame, 
                        cmd, 
                        (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 255), 
                        2
                    )
        
        return draw_frame
    
    def _add_frame_info(self, frame, frame_count, process_time, num_detections, frame_height, labels=None, frame_width=None):
        info_y_base = frame_height - 150
        
        if frame_width:
            x_position = frame_width - 200
        else:
            x_position = 10
        
        cv2.putText(
            frame, 
            f"Frame: {frame_count}", 
            (x_position, info_y_base), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 255), 
            2
        )
        
        fps_y_position = info_y_base + 30
        if process_time > 0:
            fps_text = f"Processing FPS: {1/process_time:.2f}"
        else:
            fps_text = "Processing FPS: N/A"
            
        cv2.putText(
            frame, 
            fps_text, 
            (x_position, fps_y_position), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 255), 
            2
        )
        
        if not labels:
            objects_y_position = info_y_base + 60
            cv2.putText(
                frame, 
                f"Objects: {num_detections}", 
                (x_position, objects_y_position), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 255), 
                2
            )
        
        if labels:
            label_counts = {}
            
            for label_idx in labels:
                label_idx = int(label_idx)
                if label_idx < len(self.class_names):
                    label = self.class_names[label_idx]
                    if label in label_counts:
                        label_counts[label] += 1
                    else:
                        label_counts[label] = 1
            
            top_y_base = 30
            y_offset = 0
            for label, count in label_counts.items():
                count_y_position = top_y_base + y_offset
                cv2.putText(
                    frame, 
                    f"{label}: {count}", 
                    (x_position, count_y_position), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 255), 
                    2
                )
                y_offset += 30
                    
        return frame
        
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
        if not results:
            return
            
        if 'frames' not in results:
            return
            
        if not results['frames']:
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
            
            classes = []
            count_values = []
            
            for cls, count in counts.items():
                if count > 0:
                    classes.append(cls)
                    count_values.append(count)
            
            if classes:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(classes, count_values)
                ax.set_ylabel('Count')
                ax.set_xlabel('Class')
                ax.set_title('Detections by Class')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
                
    def _get_control_commands(self, frame_center_x, frame_center_y, target_box, 
                             target_area_ratio_threshold=0.05, dead_zone_ratio=0.1):
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

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            is_inside_input_box = (
                self.input_rect[0] <= x <= self.input_rect[0] + self.input_rect[2] and
                self.input_rect[1] <= y <= self.input_rect[1] + self.input_rect[3]
            )
            
            if is_inside_input_box:
                self.input_active = True
            else:
                self.input_active = False
                
    def _apply_input_text(self):
        if not self.input_text:
            return
            
        try:
            target_id = int(self.input_text)
            
            if target_id == -1:
                self.follow_target_id = None
            elif target_id in self.available_ids:
                self.follow_target_id = target_id
            else:
                pass
                
        except ValueError:
            pass
        
        self.input_text = ""
        self.input_active = False 