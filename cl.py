import cv2
import numpy as np
import time
from ultralytics import YOLO
import argparse

def process_rtmp_feed(rtmp_url, confidence_threshold=0.5, display=True):
    """
    Process an RTMP feed with YOLO object detection
    
    Args:
        rtmp_url (str): The RTMP URL to process
        confidence_threshold (float): Confidence threshold for detections
        display (bool): Whether to display the processed feed
    """
    print(f"Connecting to RTMP feed: {rtmp_url}")
    
    # Load YOLO model (YOLOv8n by default)
    model = YOLO("yolov8n.pt")
    
    # Open RTMP stream
    cap = cv2.VideoCapture(rtmp_url)
    
    if not cap.isOpened():
        print("Error: Could not open RTMP stream.")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Stream properties: {frame_width}x{frame_height} at {fps} FPS")
    
    # Initialize video writer if needed
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    # Initialize counters for statistics
    frame_count = 0
    detection_counts = {}
    start_time = time.time()
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or error reading frame.")
            break
        
        frame_count += 1
        
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=confidence_threshold)
        
        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence
                conf = float(box.conf)
                
                # Get class id and name
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                
                # Update detection counts
                if cls_name in detection_counts:
                    detection_counts[cls_name] += 1
                else:
                    detection_counts[cls_name] = 1
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{cls_name}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display statistics on frame
        elapsed_time = time.time() - start_time
        fps_actual = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps_actual:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display top detections
        y_offset = 70
        cv2.putText(frame, "Detections:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
        
        # Sort detections by count
        sorted_detections = sorted(detection_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (cls_name, count) in enumerate(sorted_detections[:5]):  # Display top 5
            cv2.putText(frame, f"{cls_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25
        
        # Display the result frame
        if display:
            cv2.imshow('RTMP Drone Feed - Object Detection', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Write frame to output file if needed
        # out.write(frame)
    
    # Release resources
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print(f"\nProcessing completed. Total frames: {frame_count}")
    print(f"Average FPS: {frame_count / elapsed_time:.2f}")
    print("\nDetection counts:")
    for cls_name, count in sorted_detections:
        print(f"{cls_name}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RTMP stream with YOLO object detection")
    parser.add_argument("--rtmp", type=str, default="rtmp://192.168.218.70/live/stream",
                        help="RTMP URL to process")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold for detections")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not display the processed feed")
    
    args = parser.parse_args()
    
    process_rtmp_feed(args.rtmp, args.conf, not args.no_display)
