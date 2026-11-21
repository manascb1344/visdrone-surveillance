"""Guidance and mission control for autonomous target following.

The guidance controller turns a tracked target state into a body-frame
velocity command. The pilot is the mission state machine that decides
when to lock, follow, reacquire, or give up.
"""

from ..types import TargetState, VelocityCommand
from .guidance import GuidanceController
from .pilot import FollowPilot, PilotState

__all__ = [
    "FollowPilot",
    "GuidanceController",
    "PilotState",
    "TargetState",
    "VelocityCommand",
]
