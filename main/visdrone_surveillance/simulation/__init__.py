"""Simulation package for the drone follow loop.

Exposes a deterministic ground-plane world, a simulated drone camera with
projection and rendering, and a scenario that wires the two together.
"""

from __future__ import annotations

from .camera import DroneCamera
from .sim import SimScenario, StepResult
from .world import Vehicle, World

__all__ = ["DroneCamera", "SimScenario", "StepResult", "Vehicle", "World"]
