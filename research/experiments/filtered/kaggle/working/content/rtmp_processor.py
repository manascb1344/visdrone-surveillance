import os
import sys
import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO

def process_rtmp_feed(rtmp_url, model_path, confidence_threshold=0.25, display=True):
    """
    Process an RTMP feed with a VisDrone-trained YOLO model

    Args:
        rtmp_url (str): The RTMP URL to process
        model_path (str): Path to the trained YOLO model
        confidence_threshold (float): Confidence threshold for detections
        display (bool): Whether to display the processed feed
    """
    print(f"Connecting to RTMP feed: {rtmp_url}")
    print(f"Loading model from: {model_path}")

    # Load trained YOLO model
    model = YOLO(model_path)

    # VisDrone class names
    class_names = ['person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motorcycle']

    # Open RTMP stream
    # Increase buffer size for potential network latency
    # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp" # Or tcp if needed
    cap = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)


    if not cap.isOpened():
        print(f"Error: Could not open RTMP stream at {rtmp_url}")
        print("Check the URL, network connectivity, and camera stream status.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Note: FPS from live streams might be unreliable (returns 0 or default)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"Stream properties: {frame_width}x{frame_height} at {fps:.2f} FPS (Note: Live stream FPS may be inaccurate)")
    print(f"Stream properties: {frame_width}x{frame_height}")


    # Initialize counters for statistics
    frame_count = 0
    detection_counts = {class_name: 0 for class_name in class_names}
    start_time = time.time()
    last_print_time = start_time

    # Colors for visualization
    colors = [
        (0, 255, 0),    # person: green
        (255, 0, 0),    # bicycle: blue
        (0, 0, 255),    # car: red
        (255, 255, 0),  # van: cyan
        (255, 0, 255),  # truck: magenta
        (0, 255, 255),  # tricycle: yellow
        (255, 165, 0),  # bus: orange
        (128, 0, 128)   # motorcycle: purple
    ]

    # Process frames
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: End of stream or cannot read frame. Retrying connection or exiting...")
                # Optional: Add logic here to attempt reconnection after a delay
                time.sleep(2) # Wait before potentially exiting or retrying
                # Check if stream is still valid
                if not cap.isOpened():
                     print("Stream is closed. Exiting.")
                     break
                ret, frame = cap.read() # Try reading again
                if not ret:
                     print("Still cannot read frame after retry. Exiting.")
                     break # Exit if still failing

            # If frame reading is successful
            current_time = time.time()
            frame_count += 1

            # --- Run YOLO inference on the frame ---
            results = model(frame, conf=confidence_threshold, verbose=False) # Set verbose=False for less console output

            # Create a copy of the frame for drawing
            annotated_frame = frame.copy()

            # --- Process results ---
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
                    if 0 <= cls_id < len(class_names):
                         cls_name = class_names[cls_id]
                         # Update detection counts
                         if cls_name in detection_counts:
                              detection_counts[cls_name] += 1
                         color = colors[cls_id % len(colors)] # Get color for this class
                    else:
                         # Handle unexpected class ID
                         cls_name = f"class_{cls_id}"
                         color = (128, 128, 128) # Default grey color

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # Add label
                    label = f"{cls_name}: {conf:.2f}"
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # Ensure label background doesn't go out of frame top
                    label_y = max(y1, label_height + 5)
                    cv2.rectangle(annotated_frame, (x1, label_y - label_height - 5), (x1 + label_width, label_y), color, -1)
                    cv2.putText(annotated_frame, label, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


            # --- Display statistics on frame ---
            elapsed_time = current_time - start_time
            fps_actual = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Add FPS counter
            cv2.rectangle(annotated_frame, (10, 5), (150, 30), (0, 0, 0), -1) # Slightly smaller box
            cv2.putText(annotated_frame, f"FPS: {fps_actual:.1f}", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display detections counter (Top 8 classes)
            y_offset = 40 # Start below FPS counter
            max_classes_to_show = 8
            # Sort detections by count for better readability
            sorted_detections = sorted(
                [(cls_name, count) for cls_name, count in detection_counts.items() if count > 0],
                key=lambda x: x[1],
                reverse=True
            )[:max_classes_to_show] # Limit number of classes shown

            if sorted_detections:
                text_area_height = 25 + (len(sorted_detections) * 20)
                cv2.rectangle(annotated_frame, (10, y_offset), (200, y_offset + text_area_height), (0, 0, 0), -1) # Background box
                cv2.putText(annotated_frame, "Detections:", (15, y_offset + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) # Title color
                y_offset += 25 # Move down for first item

                for i, (cls_name, count) in enumerate(sorted_detections):
                    cls_id = class_names.index(cls_name) if cls_name in class_names else -1
                    color = colors[cls_id % len(colors)] if cls_id != -1 else (200, 200, 200)
                    cv2.putText(annotated_frame, f"{cls_name}: {count}", (15, y_offset + i*20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Display the result frame ---
            if display:
                # *** This is the corrected line ***
                cv2.imshow('RTMP Feed', annotated_frame)
                # Check for 'q' key press to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("'q' pressed, exiting.")
                    break
            else:
                # Print stats periodically if not displaying
                if current_time - last_print_time >= 5.0: # Print every 5 seconds
                    print(f"Processed {frame_count} frames. Current FPS: {fps_actual:.1f}")
                    last_print_time = current_time

    except KeyboardInterrupt:
        print("Stream processing interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
    finally:
        # --- Release resources ---
        print("Releasing video capture...")
        cap.release()
        if display:
            print("Destroying OpenCV display windows...")
            cv2.destroyAllWindows()

        # --- Print final statistics ---
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n--- Processing Summary ---")
        print(f"Total frames processed: {frame_count}")
        if total_time > 0:
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Average FPS: {frame_count / total_time:.2f}")
        else:
            print("Total processing time was zero.")

        print("\n--- Final Detection Counts ---")
        if any(count > 0 for count in detection_counts.values()):
             final_sorted_detections = sorted(
                  [(cls_name, count) for cls_name, count in detection_counts.items() if count > 0],
                  key=lambda x: x[1],
                  reverse=True
             )
             for cls_name, count in final_sorted_detections:
                  print(f"{cls_name}: {count}")
        else:
             print("No detections were recorded.")
        print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RTMP stream with YOLOv8 model.")
    parser.add_argument("rtmp_url", help="URL of the RTMP stream (e.g., rtmp://ip_address/live/stream)")
    parser.add_argument("model_path", help="Path to the trained YOLOv8 model (.pt file)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for detection (default: 0.3)") # Default increased slightly
    parser.add_argument("--no-display", action="store_true", help="Run without displaying the video feed")

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
         print(f"Error: Model file not found at {args.model_path}")
         sys.exit(1) # Exit if model doesn't exist

    print("Starting RTMP processing script...")
    process_rtmp_feed(args.rtmp_url, args.model_path, confidence_threshold=args.conf, display=not args.no_display)
    print("RTMP processing script finished.")

