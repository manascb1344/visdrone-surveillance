import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

class RTMPProcessor:
    """
    Class for processing RTMP feeds with a trained YOLO model.
    """
    
    def __init__(self, model_path, class_names=None):
        """
        Initialize the RTMP processor.
        
        Args:
            model_path (str): Path to the trained YOLO model
            class_names (list, optional): List of class names
        """
        self.model_path = model_path
        self.class_names = class_names or [
            'person', 'bicycle', 'car', 'van', 
            'truck', 'tricycle', 'bus', 'motorcycle'
        ]
        
        # Define colors for visualization
        self.colors = [
            (0, 255, 0),    # person: green
            (255, 0, 0),    # bicycle: blue
            (0, 0, 255),    # car: red
            (255, 255, 0),  # van: cyan
            (255, 0, 255),  # truck: magenta
            (0, 255, 255),  # tricycle: yellow
            (255, 165, 0),  # bus: orange
            (128, 0, 128)   # motorcycle: purple
        ]
        
        # Load the model
        try:
            self.model = YOLO(model_path)
            print(f"Model loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
    def install_ffmpeg(self):
        """
        Install FFmpeg for RTMP streaming
        
        Returns:
            bool: Success status
        """
        try:
            print("Installing FFmpeg...")
            os.system("apt-get update -qq && apt-get install -y ffmpeg")
            print("FFmpeg installed successfully")
            return True
        except Exception as e:
            print(f"Error installing FFmpeg: {e}")
            return False
            
    def process_feed(self, rtmp_url, output_path=None, confidence_threshold=0.25, max_frames=300):
        """
        Process an RTMP feed with the loaded YOLO model
        
        Args:
            rtmp_url (str): The RTMP URL to process
            output_path (str, optional): Path to save output video
            confidence_threshold (float): Confidence threshold for detections
            max_frames (int): Maximum frames to process
            
        Returns:
            dict: Processing statistics
        """
        if self.model is None:
            print("Error: Model not loaded.")
            return None
            
        print(f"Attempting to connect to RTMP feed: {rtmp_url}")
        
        # In non-local environments, we'll use FFmpeg to download a segment for testing
        temp_file = "/tmp/rtmp_temp.mp4"

        try:
            # Download a short segment from the RTMP stream
            print("Downloading a segment from the RTMP stream for testing...")
            os.system(f"ffmpeg -hide_banner -loglevel error -y -i {rtmp_url} -t 10 -c copy {temp_file}")

            if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                print("Failed to download RTMP stream. Check your URL and network access.")
                return None

            print(f"Successfully downloaded segment to {temp_file}")

            # Process the downloaded segment
            cap = cv2.VideoCapture(temp_file)

            if not cap.isOpened():
                print("Error: Could not open video file.")
                return None

            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"Video properties: {frame_width}x{frame_height} at {fps} FPS")

            # Set up video writer if output path is provided
            out = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # Initialize counters for statistics
            frame_count = 0
            total_detection_counts = {class_name: 0 for class_name in self.class_names} # Total counts for summary
            unique_track_ids = set() # Store unique track IDs encountered
            start_time = time.time()
            last_print_time = start_time

            # Process frames
            frames_to_display = []

            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or error reading frame.")
                    break

                frame_count += 1

                # Run YOLO inference with tracking
                # Use model.track() instead of model()
                # persist=True maintains tracks across frames
                # tracker='bytetrack.yaml' # Optionally specify tracker (botsort is default)
                results = self.model.track(frame, persist=True, conf=confidence_threshold, verbose=False)

                # Reset frame-specific counts
                frame_detection_counts = {class_name: 0 for class_name in self.class_names} 
                current_frame_track_ids = set() # Track IDs active in the current frame

                # Create a copy of the frame for drawing
                annotated_frame = frame.copy()

                # Process results
                # Check if boxes and track IDs exist in the results
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    # Combine boxes, track IDs, confidences, and classes
                    boxes = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    confs = results[0].boxes.conf.cpu().tolist()
                    clss = results[0].boxes.cls.cpu().tolist()

                    for box, track_id, conf, cls_id_float in zip(boxes, track_ids, confs, clss):
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box)
                        cls_id = int(cls_id_float)
                        
                        # Add track ID to current frame set and overall unique set
                        current_frame_track_ids.add(track_id)
                        unique_track_ids.add(track_id)

                        # Get class name
                        cls_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else f"class_{cls_id}"

                        # Update frame detection counts (by class)
                        if cls_name in frame_detection_counts:
                            frame_detection_counts[cls_name] += 1
                        # Update total detection counts (by class) - Optional if focusing on tracks
                        if cls_name in total_detection_counts:
                            total_detection_counts[cls_name] += 1

                        # Get color for this class
                        color = self.colors[cls_id % len(self.colors)]

                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                        # Add label with Track ID
                        label = f"ID:{track_id} {cls_name}: {conf:.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Display statistics on frame
                elapsed_time = time.time() - start_time
                fps_actual = frame_count / elapsed_time if elapsed_time > 0 else 0

                # Add FPS counter
                cv2.rectangle(annotated_frame, (10, 10), (200, 40), (0, 0, 0), -1)
                cv2.putText(annotated_frame, f"FPS: {fps_actual:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display detections counter (Active Tracks in Frame)
                y_offset = 50
                active_tracks_count = len(current_frame_track_ids) # Count active tracks
                cv2.rectangle(annotated_frame, (10, y_offset), (250, y_offset + 30), (0, 0, 0), -1)
                cv2.putText(annotated_frame, f"Active Tracks: {active_tracks_count}", (15, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                # Optional: Add back per-class count for the frame if needed
                # y_offset += 40 ... draw frame_detection_counts ...

                # Save frame if using output path
                if out:
                    out.write(annotated_frame)

                # Store frame for display
                if frame_count % 10 == 0 or frame_count <= 5:  # Display every 10th frame or first 5 frames
                    frames_to_display.append(annotated_frame)

            # Release resources
            cap.release()
            if out:
                out.release()
                
            # Return results
            elapsed_time = time.time() - start_time
            
            results = {
                "frame_count": frame_count,
                "elapsed_time": elapsed_time,
                "fps": frame_count / elapsed_time if elapsed_time > 0 else 0,
                "total_unique_tracks": len(unique_track_ids), # Store total unique tracks
                "total_detection_counts": total_detection_counts, # Keep class counts if needed
                "sorted_total_detections": sorted([(k,v) for k,v in total_detection_counts.items() if v > 0], key=lambda item: item[1], reverse=True),
                "frames_to_display": frames_to_display
            }
            
            return results

        except Exception as e:
            print(f"Error processing RTMP stream: {e}")
            return None

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    def display_results(self, results):
        """
        Display processing results
        
        Args:
            results (dict): Results from process_feed
        """
        if not results:
            print("No results to display.")
            return
            
        # Display selected frames
        frames = results["frames_to_display"]
        num_frames = len(frames)
        rows = (num_frames + 2) // 3  # Calculate rows needed

        plt.figure(figsize=(18, 6*rows))
        for i, frame in enumerate(frames):
            plt.subplot(rows, 3, i+1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f"Frame {i*10 if i > 5 else i}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Print final statistics
        print(f"\nProcessing completed. Total frames: {results['frame_count']}")
        print(f"Average FPS: {results['fps']:.2f}")
        print(f"Total Unique Objects Tracked: {results['total_unique_tracks']}") # Updated summary metric
        # Optional: Print total class counts if still relevant
        # print("\nTotal Detection counts by class (over all processed frames):")
        # for cls_name, count in results['sorted_total_detections']:
        #     print(f"{cls_name}: {count}")
            
    def generate_client_script(self, output_path="rtmp_processor.py"):
        """
        Generate a Python script for local RTMP processing
        
        Args:
            output_path (str): Path to save the script
            
        Returns:
            str: Path to the generated script
        """
        script_content = """
import cv2
import numpy as np
import time
import argparse
import os
import sys
from ultralytics import YOLO

def process_rtmp_feed(rtmp_url, model_path, confidence_threshold=0.25, display=True):
    \"""
    Process an RTMP feed with a VisDrone-trained YOLO model

    Args:
        rtmp_url (str): The RTMP URL to process
        model_path (str): Path to the trained YOLO model
        confidence_threshold (float): Confidence threshold for detections
        display (bool): Whether to display the processed feed
    \"""
    print(f"Connecting to RTMP feed: {rtmp_url}")
    print(f"Loading model from: {model_path}")

    # Load trained YOLO model
    model = YOLO(model_path)

    # VisDrone class names
    class_names = ['person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motorcycle']

    # Open RTMP stream
    # Increase buffer size for potential network latency
    cap = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"Error: Could not open RTMP stream at {rtmp_url}")
        print("Check the URL, network connectivity, and camera stream status.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Stream properties: {frame_width}x{frame_height}")

    # Initialize counters for statistics
    frame_count = 0
    total_detection_counts = {class_name: 0 for class_name in class_names} # Total counts for summary
    unique_track_ids = set() # Store unique track IDs encountered
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

            # --- Run YOLO inference on the frame ---\n            # Use model.track() instead of model()
            # persist=True maintains tracks across frames
            # tracker='bytetrack.yaml' # Optional
            results = model.track(frame, persist=True, conf=confidence_threshold, verbose=False) 

            # Reset frame-specific counts
            frame_detection_counts = {class_name: 0 for class_name in class_names} 
            current_frame_track_ids = set()

            # Create a copy of the frame for drawing\n            annotated_frame = frame.copy()\n
            # --- Process results ---
            # Check if boxes and track IDs exist in the results
            if results[0].boxes is not None and results[0].boxes.id is not None:
                # Combine boxes, track IDs, confidences, and classes
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()

                for box, track_id, conf, cls_id_float in zip(boxes, track_ids, confs, clss):
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box)
                    cls_id = int(cls_id_float)
                    
                    # Add track ID to sets
                    current_frame_track_ids.add(track_id)
                    unique_track_ids.add(track_id)

                    # Get class name
                    if 0 <= cls_id < len(class_names):
                         cls_name = class_names[cls_id]
                         # Update frame counts
                         if cls_name in frame_detection_counts:
                              frame_detection_counts[cls_name] += 1
                         # Update total counts
                         if cls_name in total_detection_counts:
                              total_detection_counts[cls_name] += 1
                         color = colors[cls_id % len(colors)] # Get color for this class
                    else:
                         # Handle unexpected class ID
                         cls_name = f\"class_{cls_id}\"\n                         color = (128, 128, 128) # Default grey color

                    # Draw bounding box\n                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)\n
                    # Add label with Track ID
                    label = f\"ID:{track_id} {cls_name}: {conf:.2f}\"\n                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)\n                    # Ensure label background doesn't go out of frame top\n                    label_y = max(y1, label_height + 5)\n                    cv2.rectangle(annotated_frame, (x1, label_y - label_height - 5), (x1 + label_width, label_y), color, -1)\n                    cv2.putText(annotated_frame, label, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)\n

            # --- Display statistics on frame ---
            elapsed_time = current_time - start_time
            fps_actual = frame_count / elapsed_time if elapsed_time > 0 else 0\n
            # Add FPS counter\n            cv2.rectangle(annotated_frame, (10, 5), (150, 30), (0, 0, 0), -1) # Slightly smaller box\n            cv2.putText(annotated_frame, f\"FPS: {fps_actual:.1f}\", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)\n
            # Display active tracks counter
            y_offset = 40 # Start below FPS counter
            active_tracks_count = len(current_frame_track_ids)
            cv2.rectangle(annotated_frame, (10, y_offset), (200, y_offset + 30), (0, 0, 0), -1) # Background box
            cv2.putText(annotated_frame, f\"Active Tracks: {active_tracks_count}\", (15, y_offset + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) # Title color
            # Optional: Add per-class frame counts if desired
            # y_offset += 40 ... 

            # --- Display the result frame ---
            if display:
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
            print(f"Total processing time was zero.")

        print("\n--- Final Summary ---")
        print(f"Total Unique Objects Tracked: {len(unique_track_ids)}") # Report unique tracks
        # Optional: Print total class counts
        # print("\n--- Total Detection Counts by Class ---")
        # if any(count > 0 for count in total_detection_counts.values()):
        #      final_sorted_detections = sorted(\n        #           [(cls_name, count) for cls_name, count in total_detection_counts.items() if count > 0],\n        #           key=lambda x: x[1],\n        #           reverse=True\n        #      )\n        #      for cls_name, count in final_sorted_detections:\n        #           print(f"{cls_name}: {count}")
        # else:
        #      print("No detections were recorded.")
        print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RTMP stream with YOLOv8 model.")
    parser.add_argument("rtmp_url", help="URL of the RTMP stream (e.g., rtmp://ip_address/live/stream)")
    parser.add_argument("model_path", help="Path to the trained YOLOv8 model (.pt file)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for detection (default: 0.3)")
    parser.add_argument("--no-display", action="store_true", help="Run without displaying the video feed")

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
         print(f"Error: Model file not found at {args.model_path}")
         sys.exit(1) # Exit if model doesn't exist

    print("Starting RTMP processing script...")
    process_rtmp_feed(args.rtmp_url, args.model_path, confidence_threshold=args.conf, display=not args.no_display)
    print("RTMP processing script finished.")
"""

        try:
            with open(output_path, 'w') as f:
                f.write(script_content)
            print(f"RTMP test script generated successfully at: {output_path}")
            print("\n--- Instructions for Local Usage ---")
            print(f"1. Download '{os.path.basename(output_path)}' from your system.")
            print(f"2. Download your trained model (e.g., 'visdrone_model.pt').")
            print("3. Ensure you have the necessary libraries on your local machine:")
            print("   pip install opencv-python ultralytics numpy")
            print("4. Open a terminal or command prompt on your local machine.")
            print("5. Navigate to the directory where you saved the script and model.")
            print("6. Run the script using a command like:")
            print(f"   python {os.path.basename(output_path)} rtmp://YOUR_DRONE_IP/live/stream ./visdrone_model.pt --conf 0.3")
            print("   (Replace the URL and model path with your actual values)")
            print("7. Press 'q' in the display window (if shown) or Ctrl+C in the terminal to stop.")
            print("------------------------------------")
            return output_path
            
        except Exception as e:
            print(f"Error generating RTMP script: {e}")
            return None 