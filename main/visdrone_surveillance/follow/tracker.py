"""Single-target state estimator.

The tracker owns exactly one target: the vehicle the operator picked.
It associates each incoming frame of detections with that target,
estimates image-plane velocity, and coasts on a constant-velocity model
when the target is temporarily not detected.

The tracker is a pure estimator. It performs no control and has no
knowledge of the drone, the camera, or the mission.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from ..types import BBox, Detection, TargetState

#: Minimum IoU between the predicted target box and a detection for the
#: detection to be accepted as a match.
MIN_IOU = 0.1

#: Exponential moving average weight for the velocity estimate.
VELOCITY_ALPHA = 0.5


class SingleTargetTracker:
    """Estimate the state of one tracked object across frames.

    Args:
        detection: The initial detection of the target.
        timestamp: Timestamp of the frame the detection came from.
    """

    def __init__(self, detection: Detection, timestamp: float) -> None:
        self._bbox: BBox = detection.bbox
        self._velocity: Tuple[float, float] = (0.0, 0.0)
        self._timestamp: float = timestamp
        self._missed: int = 0
        self._last_center: Optional[Tuple[float, float]] = detection.bbox.center
        self._last_match_time: Optional[float] = timestamp
        self._track_id: str = "target"

    @property
    def state(self) -> TargetState:
        """The current estimate of the target."""
        return TargetState(
            bbox=self._bbox,
            velocity=self._velocity,
            timestamp=self._timestamp,
            missed=self._missed,
            track_id=self._track_id,
        )

    @property
    def velocity(self) -> Tuple[float, float]:
        """The estimated image-plane velocity in pixels per second."""
        return self._velocity

    def lost(self, max_missed: int) -> bool:
        """Whether the target has been missing for at least ``max_missed`` frames."""
        return self._missed >= max_missed

    def update(self, detections: Iterable[Detection], timestamp: float) -> TargetState:
        """Fold a new frame of detections into the estimate.

        The target is matched against the predicted box by IoU. When no
        detection reaches the IoU threshold the nearest detection center
        within a radius proportional to the target size is used instead.
        When nothing matches, the target coasts on its last velocity and
        the missed counter grows.
        """
        frame: List[Detection] = list(detections or ())
        match = self._associate(frame, timestamp)

        if match is None:
            self._missed += 1
            self._coast(timestamp)
            return self.state

        self._missed = 0
        self._absorb(match.bbox, timestamp)
        return self.state

    def _predicted_bbox(self, timestamp: float) -> BBox:
        """The target box extrapolated with the current velocity."""
        vx, vy = self._velocity
        dt = max(0.0, timestamp - self._timestamp)
        center = self._bbox.center
        x1 = center[0] + vx * dt
        y1 = center[1] + vy * dt
        return BBox(x1, y1, x1 + self._bbox.width, y1 + self._bbox.height)

    def _associate(
        self, frame: Sequence[Detection], timestamp: float
    ) -> Optional[Detection]:
        """Choose the detection that best matches the predicted target."""
        if not frame:
            return None

        predicted = self._predicted_bbox(timestamp)
        best: Optional[Detection] = None
        best_iou = 0.0
        for det in frame:
            iou = det.bbox.iou(predicted)
            if iou >= MIN_IOU and (best is None or iou > best_iou):
                best = det
                best_iou = iou
        if best is not None:
            return best

        # Fall back to the nearest center. The search radius scales with the
        # target size so a large nearby vehicle is still reachable while a
        # small distant one does not grab detections from across the frame.
        radius = max(2.0, self._bbox.scale)
        px, py = predicted.center
        best = None
        best_dist = radius
        for det in frame:
            dx = det.bbox.center[0] - px
            dy = det.bbox.center[1] - py
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= best_dist:
                best = det
                best_dist = dist
        return best

    def _absorb(self, bbox: BBox, timestamp: float) -> None:
        """Update the estimate from a matched detection."""
        center = bbox.center
        if self._last_center is not None and self._last_match_time is not None:
            dt = timestamp - self._last_match_time
            if dt > 0:
                raw_vx = (center[0] - self._last_center[0]) / dt
                raw_vy = (center[1] - self._last_center[1]) / dt
                alpha = VELOCITY_ALPHA
                self._velocity = (
                    alpha * raw_vx + (1.0 - alpha) * self._velocity[0],
                    alpha * raw_vy + (1.0 - alpha) * self._velocity[1],
                )
        self._bbox = bbox
        self._last_center = center
        self._last_match_time = timestamp
        self._timestamp = timestamp

    def _coast(self, timestamp: float) -> None:
        """Extrapolate the target box with the stored velocity."""
        vx, vy = self._velocity
        dt = timestamp - self._timestamp
        if dt < 0:
            dt = 0.0
        x1 = self._bbox.x1 + vx * dt
        y1 = self._bbox.y1 + vy * dt
        self._bbox = BBox(x1, y1, x1 + self._bbox.width, y1 + self._bbox.height)
        self._timestamp = timestamp
