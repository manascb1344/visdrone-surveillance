"""Shared domain types for detection, tracking, and drone control.

This module is the contract between the simulator, the tracker, and the
guidance controller. Keep it free of dependencies on those packages so no
import cycle can form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("BBox corners must satisfy x2 >= x1 and y2 >= y1")

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def scale(self) -> float:
        """A single size measure, the mean of width and height.

        Guidance uses this for distance hold: a shrinking box means the
        target is getting farther away.
        """
        return (self.width + self.height) / 2.0

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass(frozen=True)
class Detection:
    """One detected object in a frame."""

    bbox: BBox
    confidence: float
    class_id: int = 0
    label: str = ""


@dataclass
class TargetState:
    """Estimated state of the single target the drone follows.

    ``velocity`` is the image-plane velocity in pixels per second. ``missed``
    counts consecutive frames with no matched detection; guidance and the
    pilot use it to decide when to coast and when to reacquire.
    """

    bbox: BBox
    velocity: Tuple[float, float] = (0.0, 0.0)
    timestamp: float = 0.0
    missed: int = 0
    track_id: str = "target"

    @property
    def center(self) -> Tuple[float, float]:
        return self.bbox.center

    @property
    def scale(self) -> float:
        return self.bbox.scale


@dataclass(frozen=True)
class VelocityCommand:
    """Body-frame command sent to the drone bridge.

    ``forward`` is meters per second along the camera heading, ``yaw_rate``
    is radians per second, ``climb`` is meters per second positive up.
    """

    forward: float = 0.0
    yaw_rate: float = 0.0
    climb: float = 0.0

    def clamped(
        self,
        max_forward: float,
        max_yaw_rate: float,
        max_climb: float,
    ) -> "VelocityCommand":
        return VelocityCommand(
            forward=max(-max_forward, min(max_forward, self.forward)),
            yaw_rate=max(-max_yaw_rate, min(max_yaw_rate, self.yaw_rate)),
            climb=max(-max_climb, min(max_climb, self.climb)),
        )
