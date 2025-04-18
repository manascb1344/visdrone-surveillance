#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for RTMP stream ingestion.
This script demonstrates the usage of the RTMP stream ingestion module.
"""

import argparse
import cv2
import logging
import time
import sys
import signal
from typing import Dict, Any

# Import our RTMP modules
from stream_ingestion import VisDroneRTMPClient, RTMPStreamManager

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

def display_stream(client: VisDroneRTMPClient, window_name: str = "RTMP Stream") -> None:
    """
    Display the RTMP stream in a window using OpenCV.
    
    Args:
        client: The RTMP client to get frames from
        window_name: Name of the display window
    """
    global running
    
    # Register for stream events
    client.register_event_callback("connect", log_event)
    client.register_event_callback("disconnect", log_event)
    
    # Create window and start client
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    
    if not client.start():
        logger.error("Failed to start RTMP stream")
        return
    
    logger.info(f"Displaying stream: {client.rtmp_url}")
    
    try:
        last_stats_time = time.time()
        
        while running:
            # Get latest frame
            frame = client.get_latest_frame()
            
            if frame is not None:
                # Display FPS and other info on frame
                stats = client.get_stats()
                fps_text = f"FPS: {stats['fps']:.1f}"
                dropped_text = f"Dropped: {stats['dropped_frames']}"
                buffer_text = f"Buffer: {stats['buffer_usage']}/{stats['buffer_capacity']}"
                
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2)
                cv2.putText(frame, dropped_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2)
                cv2.putText(frame, buffer_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow(window_name, frame)
            
            # Print stats every 5 seconds
            current_time = time.time()
            if current_time - last_stats_time > 5.0:
                stats = client.get_stats()
                logger.info(f"Stream stats: FPS={stats['fps']:.1f}, "
                            f"Dropped={stats['dropped_frames']}, "
                            f"Buffer={stats['buffer_usage']}/{stats['buffer_capacity']}")
                last_stats_time = current_time
            
            # Check for key press (q or ESC to quit)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                running = False
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error in display loop: {e}")
    finally:
        client.stop()
        cv2.destroyAllWindows()

def test_multi_stream(rtmp_urls: list, buffer_size: int = 30) -> None:
    """
    Test multiple RTMP streams using the RTMPStreamManager.
    
    Args:
        rtmp_urls: List of RTMP URLs to test
        buffer_size: Size of the frame buffer for each stream
    """
    global running
    
    manager = RTMPStreamManager()
    
    # Add all streams
    for i, url in enumerate(rtmp_urls):
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
                
                # Create a blank canvas
                grid = create_grid_display(frames, rows, cols, 
                                          frame_width, frame_height)
                
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
    # Create a blank canvas
    grid = cv2.cvtColor(
        cv2.imread("../../datasets/placeholder.jpg") 
        if cv2.imread("../../datasets/placeholder.jpg") is not None
        else cv2.imread("/dev/null"),  # Will be None, creating a black image
        cv2.COLOR_BGR2RGB
    )
    
    if grid is None:
        # Create a black image if placeholder.jpg doesn't exist
        grid = cv2.cvtColor(
            255 * np.ones((rows * frame_height, cols * frame_width, 3), dtype=np.uint8),
            cv2.COLOR_RGB2BGR
        )
    else:
        # Resize placeholder to the grid size
        grid = cv2.resize(grid, (cols * frame_width, rows * frame_height))
    
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

def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test RTMP Stream Ingestion")
    
    parser.add_argument(
        "rtmp_url", 
        help="RTMP URL to stream from. For multiple streams, separate with commas."
    )
    
    parser.add_argument(
        "--buffer-size", type=int, default=30,
        help="Size of the frame buffer (default: 30)"
    )
    
    parser.add_argument(
        "--multi", action="store_true",
        help="Enable multi-stream view for multiple RTMP URLs"
    )
    
    args = parser.parse_args()
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse comma-separated RTMP URLs
    rtmp_urls = [url.strip() for url in args.rtmp_url.split(",")]
    
    if args.multi or len(rtmp_urls) > 1:
        logger.info(f"Testing multi-stream with {len(rtmp_urls)} URLs")
        test_multi_stream(rtmp_urls, args.buffer_size)
    else:
        # Single stream test
        client = VisDroneRTMPClient(rtmp_urls[0], args.buffer_size)
        display_stream(client)
    
    logger.info("Test completed")

if __name__ == "__main__":
    # Add missing import for create_grid_display
    import numpy as np
    main() 