#!/usr/bin/env python

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.train import train, evaluate_model

