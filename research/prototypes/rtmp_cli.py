#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for RTMP stream handling.
This script provides a convenient interface to work with RTMP streams.
"""

import argparse
import cv2
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import our RTMP modules
from rtmp import VisDroneRTMPClient, RTMPStreamManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) to gracefully shutdown."""
    global running
    logger.info("Shutting down...")
    running = False

def log_event(event_data: Dict[str, Any]) -> None:
    """Event callback for logging stream events."""
    logger.info(f"Stream event: {event_data}")

def display_stream(stream_url: str, buffer_size: int = 30, save_path: Optional[str] = None, 
                  save_interval: int = 0, display: bool = True) -> None:
    """
    Display and/or save frames from an RTMP stream.
    
    Args:
        stream_url: URL of the RTMP stream
        buffer_size: Size of the frame buffer (in frames)
        save_path: Directory to save frames (None = no saving)
        save_interval: Interval in seconds between saved frames (0 = save all frames)
        display: Whether to display the stream in a window
    """
    global running
    
    # Create client
    client = VisDroneRTMPClient(stream_url, buffer_size)
    
    # Register for stream events
    client.register_event_callback("connect", log_event)
    client.register_event_callback("disconnect", log_event)
    
    # Create display window if needed
    if display:
        window_name = "RTMP Stream"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)
    
    # Create save directory if needed
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving frames to {save_path}")
        last_save_time = 0
    
    # Start client
    if not client.start():
        logger.error("Failed to start RTMP stream")
        return
    
    logger.info(f"Connected to stream: {client.rtmp_url}")
    
    try:
        last_stats_time = time.time()
        
        while running:
            # Get latest frame
            frame = client.get_latest_frame()
            
            if frame is not None:
                # Display frame if enabled
                if display:
                    # Display FPS and other info on frame
                    stats = client.get_stats()
                    fps_text = f"FPS: {stats['fps']:.1f}"
                    dropped_text = f"Dropped: {stats['dropped_frames']}"
                    buffer_text = f"Buffer: {stats['buffer_usage']}/{stats['buffer_capacity']}"
                    
                    overlay_frame = frame.copy()
                    cv2.putText(overlay_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2)
                    cv2.putText(overlay_frame, dropped_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2)
                    cv2.putText(overlay_frame, buffer_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2)
                    
                    cv2.imshow(window_name, overlay_frame)
                
                # Save frame if enabled
                if save_path:
                    current_time = time.time()
                    if (save_interval == 0) or (current_time - last_save_time >= save_interval):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = os.path.join(save_path, f"frame_{timestamp}.jpg")
                        cv2.imwrite(filename, frame)
                        last_save_time = current_time
            
            # Print stats every 5 seconds
            current_time = time.time()
            if current_time - last_stats_time > 5.0:
                stats = client.get_stats()
                logger.info(f"Stream stats: FPS={stats['fps']:.1f}, "
                           f"Dropped={stats['dropped_frames']}, "
                           f"Buffer={stats['buffer_usage']}/{stats['buffer_capacity']}")
                last_stats_time = current_time
            
            # Check for key press if display is enabled
            if display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    running = False
                    break
                elif key == ord('s') and save_path:  # 's' to save a frame manually
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(save_path, f"manual_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    logger.info(f"Frame manually saved to {filename}")
            else:
                # Small sleep to avoid CPU hogging when not displaying
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error in stream handling: {e}")
    finally:
        client.stop()
        if display:
            cv2.destroyAllWindows()

def list_streams(stream_urls: List[str]) -> None:
    """
    Display information about RTMP streams without connecting fully.
    
    Args:
        stream_urls: List of RTMP stream URLs to check
    """
    logger.info(f"Checking {len(stream_urls)} RTMP streams...")
    
    for i, url in enumerate(stream_urls):
        logger.info(f"Stream {i+1}: {url}")
        
        # Create a temporary client to get info
        client = VisDroneRTMPClient(url)
        
        try:
            # Try to connect
            if client.connect():
                # Get a frame to ensure stream is working
                success, _ = client.read_frame()
                
                if success:
                    # Get stream info
                    info = client.get_stream_info()
                    logger.info(f"  Status: Connected")
                    logger.info(f"  Resolution: {info.get('width', 'Unknown')}x{info.get('height', 'Unknown')}")
                    logger.info(f"  FPS: {info.get('fps', 'Unknown')}")
                else:
                    logger.warning(f"  Status: Connected but failed to read frames")
            else:
                logger.warning(f"  Status: Connection failed")
                
        except Exception as e:
            logger.error(f"  Error checking stream: {e}")
        finally:
            # Always disconnect
            client.disconnect()
            
        logger.info("------------------------------")

def run_multi_stream_view(stream_urls: List[str], buffer_size: int = 30) -> None:
    """
    Run a multi-stream viewer with the given RTMP URLs.
    
    Args:
        stream_urls: List of RTMP stream URLs to display
        buffer_size: Size of the frame buffer for each stream
    """
    global running
    
    # Create stream manager
    manager = RTMPStreamManager()
    
    # Add all streams
    for i, url in enumerate(stream_urls):
        stream_id = f"stream_{i}"
        logger.info(f"Adding stream {stream_id}: {url}")
        
        success = manager.add_stream(stream_id, url, buffer_size)
        if not success:
            logger.warning(f"Failed to add stream {stream_id}")
    
    if not manager.get_active_streams():
        logger.error("No active streams")
        return
    
    logger.info(f"Active streams: {manager.get_active_streams()}")
    
    # Create display window
    cv2.namedWindow("Multi-Stream View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi-Stream View", 1280, 720)
    
    try:
        last_stats_time = time.time()
        
        while running:
            # Get latest frames from all streams
            frames = manager.get_latest_frames()
            
            if frames:
                # Create a grid of frames
                rows = (len(frames) + 1) // 2  # Ceiling division
                cols = min(2, len(frames))
                
                # Calculate frame size to fit in a grid
                frame_height = 720 // rows
                frame_width = 1280 // cols
                
                # Create a grid display
                grid = create_grid_display(frames, rows, cols, frame_width, frame_height)
                
                # Show the grid
                cv2.imshow("Multi-Stream View", grid)
            
            # Print stats every 5 seconds
            current_time = time.time()
            if current_time - last_stats_time > 5.0:
                all_stats = manager.get_all_stats()
                for stream_id, stats in all_stats.items():
                    logger.info(f"Stream {stream_id} stats: "
                                f"FPS={stats['fps']:.1f}, "
                                f"Dropped={stats['dropped_frames']}")
                last_stats_time = current_time
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                running = False
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error in multi-stream display: {e}")
    finally:
        manager.stop_all_streams()
        cv2.destroyAllWindows()

def create_grid_display(frames: Dict[str, cv2.typing.MatLike], rows: int, cols: int, 
                       frame_width: int, frame_height: int) -> cv2.typing.MatLike:
    """
    Create a grid display of multiple video frames.
    
    Args:
        frames: Dictionary of stream_id -> frame
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        frame_width: Width of each frame in the grid
        frame_height: Height of each frame in the grid
        
    Returns:
        cv2.typing.MatLike: Grid image with all frames
    """
    import numpy as np
    
    # Create a black image as the base
    grid = np.zeros((rows * frame_height, cols * frame_width, 3), dtype=np.uint8)
    
    # Place each frame in the grid
    for i, (stream_id, frame) in enumerate(frames.items()):
        if i >= rows * cols:
            break  # Skip if we have more frames than grid cells
            
        row = i // cols
        col = i % cols
        
        # Resize the frame to fit in the grid
        resized_frame = cv2.resize(frame, (frame_width, frame_height))
        
        # Calculate position
        y_start = row * frame_height
        y_end = y_start + frame_height
        x_start = col * frame_width
        x_end = x_start + frame_width
        
        # Place frame in the grid
        grid[y_start:y_end, x_start:x_end] = resized_frame
        
        # Add stream ID label
        cv2.putText(grid, stream_id, (x_start + 10, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return grid

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RTMP Stream CLI")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Display stream command
    display_parser = subparsers.add_parser("display", help="Display an RTMP stream")
    display_parser.add_argument("stream_url", help="RTMP URL to stream from")
    display_parser.add_argument("--buffer-size", type=int, default=30,
                              help="Size of the frame buffer (default: 30)")
    display_parser.add_argument("--save-path", 
                              help="Directory to save frames (if not provided, frames are not saved)")
    display_parser.add_argument("--save-interval", type=int, default=0,
                              help="Interval in seconds between saved frames (0 = save all frames)")
    
    # List streams command
    list_parser = subparsers.add_parser("list", help="List information about RTMP streams")
    list_parser.add_argument("stream_urls", nargs="+", help="RTMP URLs to check")
    
    # Multi-stream command
    multi_parser = subparsers.add_parser("multi", help="Display multiple RTMP streams")
    multi_parser.add_argument("stream_urls", nargs="+", help="RTMP URLs to stream from")
    multi_parser.add_argument("--buffer-size", type=int, default=30,
                            help="Size of the frame buffer (default: 30)")
    
    return parser.parse_args()

def main():
    """Main function to run the CLI."""
    args = parse_args()
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Handle commands
    if args.command == "display":
        display_stream(args.stream_url, args.buffer_size, args.save_path, args.save_interval)
    elif args.command == "list":
        list_streams(args.stream_urls)
    elif args.command == "multi":
        run_multi_stream_view(args.stream_urls, args.buffer_size)
    else:
        logger.error("No command specified. Use --help for usage information.")
    
    logger.info("Exiting...")

if __name__ == "__main__":
    main() 