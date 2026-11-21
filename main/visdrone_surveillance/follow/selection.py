"""Turn an operator pick into a single target detection.

Both functions are pure: they take detections and a pick and return a
detection or ``None``. They never mutate their inputs.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..types import BBox, Detection


def _contains(bbox: BBox, x: float, y: float) -> bool:
    return bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2


def select_by_point(
    detections: Iterable[Detection],
    x: float,
    y: float,
) -> Optional[Detection]:
    """Pick the detection whose bbox contains the clicked pixel.

    When several boxes overlap the click (a car inside a truck trailer,
    for instance) the smallest box wins, since the operator most likely
    aimed at the smaller object. Returns ``None`` when the click misses
    every detection or the input is empty.
    """
    best: Optional[Detection] = None
    for det in detections or ():
        if not _contains(det.bbox, x, y):
            continue
        if best is None or det.bbox.area < best.bbox.area:
            best = det
    return best


def select_by_bbox(
    detections: Iterable[Detection],
    bbox: BBox,
    min_iou: float = 0.3,
) -> Optional[Detection]:
    """Pick the most confident detection overlapping the given box.

    The box comes from an operator dragging a rectangle in the video
    feed. Among detections whose IoU with it reaches ``min_iou`` the
    highest confidence one wins. Returns ``None`` when nothing overlaps
    enough or the input is empty.
    """
    best: Optional[Detection] = None
    for det in detections or ():
        if det.bbox.iou(bbox) < min_iou:
            continue
        if best is None or det.confidence > best.confidence:
            best = det
    return best
