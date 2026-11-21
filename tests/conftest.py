"""Characterization tests pinning current behavior before the restructure.

These assert observable behavior of the existing modules, not their
internals. They are the equivalence harness for the move to
``main/visdrone_surveillance/``: the same assertions must hold after.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "main"))
sys.path.insert(0, ROOT)
