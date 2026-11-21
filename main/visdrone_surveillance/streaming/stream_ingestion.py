#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RTMP Stream Ingestion implementation for VisDrone monitoring.
This module provides concrete implementations of RTMPClient for video ingestion.
"""

import cv2
import logging
import time
import queue
import threading
from typing import Optional, List, Dict, Any, Callable

from .rtmp_client import RTMPClient

logger = logging.getLogger(__name__)

class VisDroneRTMPClient(RTMPClient):
    """
    RTMP client implementation for VisDrone monitoring.
    Adds frame buffering, statistics tracking, and event handling.
    """
    
    def __init__(self, rtmp_url: str, buffer_size: int = 30, 
                 reconnect_attempts: int = 5, reconnect_delay: float = 3.0):
        """
        Initialize the VisDrone RTMP client.
        
        Args:
            rtmp_url: The URL of the RTMP stream
            buffer_size: Size of the frame buffer (in frames)
            reconnect_attempts: Number of reconnection attempts if connection fails
            reconnect_delay: Delay in seconds between reconnection attempts
        """
        super().__init__(rtmp_url, reconnect_attempts, reconnect_delay)
        self.buffer_size = buffer_size
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.frame_timestamps = []
        self.stats = {
            "fps": 0.0,
            "dropped_frames": 0,
            "reconnections": 0,
            "uptime": 0.0,
            "last_activity": 0.0
        }
        self.start_time = 0.0
        self.event_callbacks = {}
        self.frame_lock = threading.Lock()
        self.is_running = False
        self.thread = None
    
    def on_connect(self) -> None:
        """Called when connection to RTMP stream is established."""
        self.start_time = time.time()
        self.stats["last_activity"] = self.start_time
        self._trigger_event("connect", {"url": self.rtmp_url, "time": self.start_time})
    
    def on_disconnect(self) -> None:
        """Called when disconnected from RTMP stream."""
        self.stats["uptime"] = time.time() - self.start_time
        self._trigger_event("disconnect", {"url": self.rtmp_url, "uptime": self.stats["uptime"]})
    
    def on_frame(self, frame: cv2.typing.MatLike) -> None:
        """
        Process a new frame from the stream.
        
        Args:
            frame: The received video frame
        """
        current_time = time.time()
        self.stats["last_activity"] = current_time
        
        # Update FPS calculation
        self.frame_timestamps.append(current_time)
        # Keep only last 100 timestamps for FPS calculation
        if len(self.frame_timestamps) > 100:
            self.frame_timestamps = self.frame_timestamps[-100:]
        
        if len(self.frame_timestamps) >= 2:
            time_diff = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if time_diff > 0:
                self.stats["fps"] = (len(self.frame_timestamps) - 1) / time_diff
        
        # Add frame to buffer, dropping oldest if full
        try:
            if self.frame_buffer.full():
                self.frame_buffer.get_nowait()
                self.stats["dropped_frames"] += 1
            
            self.frame_buffer.put_nowait(frame.copy())
            self._trigger_event("frame", {"frame_count": self.frame_count})
            
        except queue.Full:
            self.stats["dropped_frames"] += 1
    
    def start(self) -> bool:
        """
        Start ingesting the RTMP stream in a background thread.
        
        Returns:
            bool: True if successfully started, False otherwise
        """
        if self.is_running:
            logger.warning("Stream ingestion already running")
            return True
        
        if self.connect():
            self.is_running = True
            self.thread = threading.Thread(target=self._ingest_stream, daemon=True)
            self.thread.start()
            return True
        
        return False
    
    def stop(self) -> None:
        """Stop ingesting the RTMP stream."""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=3.0)
            self.thread = None
        
        self.disconnect()
    
    def _ingest_stream(self) -> None:
        """Background thread function for ingesting stream frames."""
        while self.is_running and self.is_connected:
            success, _ = self.read_frame()  # Frame is already processed in on_frame
            if not success:
                # Small delay to prevent CPU spinning if reads are failing
                time.sleep(0.01)
    
    def get_latest_frame(self) -> Optional[cv2.typing.MatLike]:
        """
        Get the most recent frame from the buffer without removing it.
        
        Returns:
            Optional[cv2.typing.MatLike]: The latest frame or None if buffer is empty
        """
        if self.frame_buffer.empty():
            return None
        
        with self.frame_lock:
            # Get all frames, but only keep the latest
            frames = []
            try:
                while not self.frame_buffer.empty():
                    frames.append(self.frame_buffer.get_nowait())
            except queue.Empty:
                pass
            
            # Put all frames except the last one back
            for frame in frames[:-1]:
                try:
                    self.frame_buffer.put_nowait(frame)
                except queue.Full:
                    # If buffer is full, just drop frames
                    self.stats["dropped_frames"] += 1
            
            return frames[-1] if frames else None
    
    def get_frame_batch(self, batch_size: int = 1) -> List[cv2.typing.MatLike]:
        """
        Get a batch of frames from the buffer.
        
        Args:
            batch_size: Number of frames to retrieve
            
        Returns:
            List[cv2.typing.MatLike]: List of frames
        """
        frames = []
        with self.frame_lock:
            for _ in range(min(batch_size, self.frame_buffer.qsize())):
                try:
                    frames.append(self.frame_buffer.get_nowait())
                except queue.Empty:
                    break
        
        return frames
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics for the stream ingestion.
        
        Returns:
            Dict[str, Any]: Dictionary with streaming statistics
        """
        stats = self.stats.copy()
        stats.update({
            "buffer_usage": self.frame_buffer.qsize(),
            "buffer_capacity": self.buffer_size,
            "stream_info": self.get_stream_info()
        })
        return stats
    
    def register_event_callback(self, event_name: str, callback: Callable) -> None:
        """
        Register a callback function for specific events.
        
        Args:
            event_name: Name of the event to register for
            callback: Function to call when event occurs
        """
        if event_name not in self.event_callbacks:
            self.event_callbacks[event_name] = []
        
        self.event_callbacks[event_name].append(callback)
    
    def _trigger_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """
        Trigger callbacks for a specific event.
        
        Args:
            event_name: Name of the event
            event_data: Data associated with the event
        """
        if event_name in self.event_callbacks:
            for callback in self.event_callbacks[event_name]:
                try:
                    callback(event_data)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")


class RTMPStreamManager:
    """
    Manages multiple RTMP streams.
    Provides a single interface to handle multiple camera feeds.
    """
    
    def __init__(self):
        """Initialize the RTMP stream manager."""
        self.streams = {}  # Dict of VisDroneRTMPClient instances by stream_id
    
    def add_stream(self, stream_id: str, rtmp_url: str, buffer_size: int = 30) -> bool:
        """
        Add a new RTMP stream to the manager.
        
        Args:
            stream_id: Unique identifier for the stream
            rtmp_url: RTMP URL for the stream
            buffer_size: Frame buffer size for the stream
            
        Returns:
            bool: True if stream was added successfully, False otherwise
        """
        if stream_id in self.streams:
            logger.warning(f"Stream ID '{stream_id}' already exists, replacing")
            self.remove_stream(stream_id)
        
        client = VisDroneRTMPClient(rtmp_url, buffer_size)
        self.streams[stream_id] = client
        
        return client.start()
    
    def remove_stream(self, stream_id: str) -> None:
        """
        Remove an RTMP stream from the manager.
        
        Args:
            stream_id: ID of the stream to remove
        """
        if stream_id in self.streams:
            self.streams[stream_id].stop()
            del self.streams[stream_id]
    
    def get_stream(self, stream_id: str) -> Optional[VisDroneRTMPClient]:
        """
        Get a stream client by ID.
        
        Args:
            stream_id: ID of the stream to retrieve
            
        Returns:
            Optional[VisDroneRTMPClient]: The stream client or None if not found
        """
        return self.streams.get(stream_id)
    
    def get_latest_frames(self) -> Dict[str, cv2.typing.MatLike]:
        """
        Get the latest frame from each stream.
        
        Returns:
            Dict[str, cv2.typing.MatLike]: Dictionary mapping stream IDs to frames
        """
        frames = {}
        for stream_id, client in self.streams.items():
            frame = client.get_latest_frame()
            if frame is not None:
                frames[stream_id] = frame
        
        return frames
    
    def get_active_streams(self) -> List[str]:
        """
        Get list of active stream IDs.
        
        Returns:
            List[str]: List of active stream IDs
        """
        return list(self.streams.keys())
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all streams.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping stream IDs to stats
        """
        return {stream_id: client.get_stats() for stream_id, client in self.streams.items()}
    
    def stop_all_streams(self) -> None:
        """Stop all active streams."""
        for client in self.streams.values():
            client.stop()
        
        self.streams.clear() 