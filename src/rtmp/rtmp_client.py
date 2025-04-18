#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RTMP Client module for connecting to and reading from RTMP streams.
This module provides a base class for RTMP stream ingestion.
"""

import cv2
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RTMPClient(ABC):
    """Base class for RTMP stream ingestion."""
    
    def __init__(self, rtmp_url: str, reconnect_attempts: int = 5, 
                 reconnect_delay: float = 3.0):
        """
        Initialize the RTMP client.
        
        Args:
            rtmp_url: The URL of the RTMP stream
            reconnect_attempts: Number of reconnection attempts if connection fails
            reconnect_delay: Delay in seconds between reconnection attempts
        """
        self.rtmp_url = rtmp_url
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.is_connected = False
        self.cap = None
        self.frame_count = 0
        self.last_reconnect_time = 0
        
    def connect(self) -> bool:
        """
        Connect to the RTMP stream.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        logger.info(f"Connecting to RTMP stream: {self.rtmp_url}")
        
        attempts = 0
        while attempts < self.reconnect_attempts:
            try:
                self.cap = cv2.VideoCapture(self.rtmp_url)
                if not self.cap.isOpened():
                    logger.error(f"Failed to open RTMP stream: {self.rtmp_url}")
                    attempts += 1
                    time.sleep(self.reconnect_delay)
                    continue
                
                self.is_connected = True
                self.on_connect()
                logger.info(f"Successfully connected to RTMP stream: {self.rtmp_url}")
                return True
                
            except Exception as e:
                logger.error(f"Error connecting to RTMP stream: {e}")
                attempts += 1
                if attempts < self.reconnect_attempts:
                    logger.info(f"Retrying connection in {self.reconnect_delay} seconds...")
                    time.sleep(self.reconnect_delay)
        
        logger.error(f"Failed to connect after {self.reconnect_attempts} attempts")
        return False
    
    def disconnect(self) -> None:
        """Disconnect from the RTMP stream."""
        if self.is_connected and self.cap is not None:
            self.cap.release()
            self.is_connected = False
            self.on_disconnect()
            logger.info(f"Disconnected from RTMP stream: {self.rtmp_url}")
    
    def read_frame(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        """
        Read a frame from the RTMP stream.
        
        Returns:
            tuple: (success, frame) where success is a boolean indicating if the read was
                  successful and frame is the image frame (None if not successful)
        """
        if not self.is_connected or self.cap is None:
            logger.error("Cannot read frame: Not connected to stream")
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            self.on_frame(frame)
            return True, frame
        else:
            logger.warning("Failed to read frame, attempting to reconnect...")
            self.handle_reconnection()
            return False, None
    
    def stream(self) -> Generator[cv2.typing.MatLike, None, None]:
        """
        Generator that yields frames from the RTMP stream.
        
        Yields:
            cv2.typing.MatLike: Video frames from the stream
        """
        if not self.is_connected and not self.connect():
            logger.error("Could not start streaming: Failed to connect")
            return
        
        while self.is_connected:
            success, frame = self.read_frame()
            if success and frame is not None:
                yield frame
            else:
                # If we can't read a frame and reconnection failed, we're done
                if not self.is_connected:
                    break
    
    def handle_reconnection(self) -> bool:
        """
        Handle reconnection logic when connection to stream is lost.
        
        Returns:
            bool: True if reconnection successful, False otherwise
        """
        current_time = time.time()
        # Avoid excessive reconnection attempts
        if current_time - self.last_reconnect_time < self.reconnect_delay:
            time.sleep(self.reconnect_delay)
            
        self.last_reconnect_time = current_time
        self.disconnect()
        return self.connect()
    
    def get_stream_info(self) -> dict:
        """
        Get information about the connected stream.
        
        Returns:
            dict: Dictionary with stream information
        """
        if not self.is_connected or self.cap is None:
            return {
                "connected": False,
                "url": self.rtmp_url,
                "frame_count": self.frame_count
            }
            
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        return {
            "connected": True,
            "url": self.rtmp_url,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": self.frame_count
        }
    
    @abstractmethod
    def on_connect(self) -> None:
        """Called when connection to RTMP stream is established."""
        pass
    
    @abstractmethod
    def on_disconnect(self) -> None:
        """Called when disconnected from RTMP stream."""
        pass
    
    @abstractmethod
    def on_frame(self, frame: cv2.typing.MatLike) -> None:
        """
        Called when a new frame is received from the stream.
        
        Args:
            frame: The received video frame
        """
        pass 