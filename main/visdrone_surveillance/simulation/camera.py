"""Simulated drone camera: pose, projection, rendering, and observation.

The camera model is a pinhole camera looking straight down at the ground
plane. Ground points are projected orthographically onto the image plane with
a scale factor derived from the altitude and the vertical field of view:

    f = (height / 2) / tan(vfov / 2)      pixels
    pixels_per_meter = f / altitude

The image plane is rotated by the camera yaw. A point ahead of the drone
appears near the top of the image.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from ..types import BBox, Detection, VelocityCommand
from .world import World

# Background color of the rendered asphalt frame, in BGR.
ASPHALT_COLOR: Tuple[int, int, int] = (38, 36, 34)


class DroneCamera:
    """Camera pose on the drone plus the projection and rendering helpers."""

    def __init__(
        self,
        position: Tuple[float, float, float] = (0.0, 0.0, 20.0),
        yaw: float = 0.0,
        image_width: int = 640,
        image_height: int = 480,
        vfov_deg: float = 60.0,
    ) -> None:
        """Create the camera.

        Args:
            position: ``(x, y, altitude)`` in meters.
            yaw: Heading of the camera in radians measured from the +x axis.
            image_width: Frame width in pixels.
            image_height: Frame height in pixels.
            vfov_deg: Vertical field of view in degrees.
        """
        self.x = float(position[0])
        self.y = float(position[1])
        self.z = float(position[2])
        self.yaw = float(yaw)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.vfov_deg = float(vfov_deg)

    # ------------------------------------------------------------------
    # Pose integration
    # ------------------------------------------------------------------
    def apply(self, command: VelocityCommand, dt: float) -> None:
        """Integrate a body-frame command into the camera pose.

        Forward moves along the current heading, ``yaw_rate`` rotates the
        heading, and ``climb`` changes the altitude. No clamping is done
        here, the controller is responsible for that.
        """
        self.yaw += command.yaw_rate * dt
        self.x += command.forward * math.cos(self.yaw) * dt
        self.y += command.forward * math.sin(self.yaw) * dt
        self.z += command.climb * dt

    @property
    def position(self) -> Tuple[float, float, float]:
        """Current ``(x, y, altitude)`` of the camera."""
        return (self.x, self.y, self.z)

    def set_position(self, position: Sequence[float]) -> None:
        """Set the camera pose from an ``(x, y, altitude)`` triple."""
        self.x = float(position[0])
        self.y = float(position[1])
        self.z = float(position[2])

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    @property
    def focal_length(self) -> float:
        """Focal length in pixels derived from the image height and vfov."""
        return (self.image_height / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)

    @property
    def pixels_per_meter(self) -> float:
        """Ground scale of the projection in pixels per meter."""
        if self.z <= 1e-6:
            return 0.0
        return self.focal_length / self.z

    def world_to_image(self, pos: Sequence[float]) -> Tuple[float, float]:
        """Project a ground point to pixel coordinates.

        Args:
            pos: ``(x, y)`` ground position in meters.

        Returns:
            ``(u, v)`` pixel coordinates. The point may fall outside the frame.
        """
        dx = pos[0] - self.x
        dy = pos[1] - self.y
        scale = self.pixels_per_meter
        # Camera frame axes on the ground plane: forward along the heading,
        # right perpendicular to it.
        forward = dx * math.cos(self.yaw) + dy * math.sin(self.yaw)
        right = -dx * math.sin(self.yaw) + dy * math.cos(self.yaw)
        u = self.image_width / 2.0 + right * scale
        v = self.image_height / 2.0 - forward * scale
        return (u, v)

    def image_to_world(self, uv: Sequence[float]) -> Tuple[float, float]:
        """Inverse of :meth:`world_to_image` for a ground point."""
        scale = self.pixels_per_meter
        right = (uv[0] - self.image_width / 2.0) / scale
        forward = -(uv[1] - self.image_height / 2.0) / scale
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        return (self.x + forward * c - right * s, self.y + forward * s + right * c)

    # ------------------------------------------------------------------
    # Rendering and observation
    # ------------------------------------------------------------------
    def render(self, world: World) -> np.ndarray:
        """Render a top-down orthographic view of the world.

        Args:
            world: The world whose vehicles are drawn.

        Returns:
            An ``HxWx3`` uint8 BGR frame.
        """
        frame = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        frame[:, :] = ASPHALT_COLOR
        for v in world.vehicles:
            pts = np.array(
                [self.world_to_image(c) for c in v.corners()], dtype=np.float64
            )
            poly = np.round(pts).astype(np.int32)
            cv2.fillConvexPoly(frame, poly, v.color)
            cv2.polylines(frame, [poly], True, (20, 20, 20), 1)
        return frame

    def observe(self, world: World) -> List[Detection]:
        """Return ground truth detections for every vehicle touching the frame.

        The bounding box is the axis-aligned hull of the projected vehicle
        corners, clipped to the image, which matches what :meth:`render`
        draws.
        """
        detections: List[Detection] = []
        for v in world.vehicles:
            pts = [self.world_to_image(c) for c in v.corners()]
            us = [p[0] for p in pts]
            vs = [p[1] for p in pts]
            x1 = max(0.0, min(us))
            y1 = max(0.0, min(vs))
            x2 = min(float(self.image_width), max(us))
            y2 = min(float(self.image_height), max(vs))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    bbox=BBox(x1, y1, x2, y2),
                    confidence=1.0,
                    class_id=0,
                    label="vehicle",
                )
            )
        return detections
