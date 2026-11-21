"""Target selection and single-target tracking.

The operator picks an object in the video feed; these modules turn that
pick into a tracked target and an estimated target state.
"""

from ..types import BBox, Detection, TargetState
from .selection import select_by_bbox, select_by_point
from .tracker import SingleTargetTracker

__all__ = [
    "BBox",
    "Detection",
    "SingleTargetTracker",
    "TargetState",
    "select_by_bbox",
    "select_by_point",
]
