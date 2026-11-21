#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RTMP streaming module for VisDrone object detection pipeline.
This package provides functionality for ingesting and processing RTMP video streams.
"""

from .handler import RTMPHandler
from .rtmp_client import RTMPClient
from .stream_ingestion import RTMPStreamManager, VisDroneRTMPClient

__all__ = [
    "RTMPClient",
    "RTMPHandler",
    "RTMPStreamManager",
    "VisDroneRTMPClient",
]

__version__ = "0.1.0"
