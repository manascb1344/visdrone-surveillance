#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RTMP streaming module for VisDrone object detection pipeline.
This package provides functionality for ingesting and processing RTMP video streams.
"""

from .rtmp_client import RTMPClient
from .stream_ingestion import VisDroneRTMPClient, RTMPStreamManager

__all__ = ['RTMPClient', 'VisDroneRTMPClient', 'RTMPStreamManager']

__version__ = '0.1.0' 