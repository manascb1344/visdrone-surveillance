"""Pin the behavior of tracking and speed estimation.

These modules carry real logic (trajectory prediction, speed smoothing,
calibration) and are the parts of the surveillance stack most likely to
change in a restructure, so they are pinned here.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visdrone_surveillance.surveillance.vehicle_tracking import (
    TrackingPoint,
    VehicleTrack,
    MultiCameraTracker,
)
from visdrone_surveillance.surveillance.speed_estimation import SpeedEstimator


def _point(x, y, camera_id="cam1", timestamp=0.0, frame_id=0):
    return TrackingPoint(
        vehicle_id="v1",
        frame_id=frame_id,
        camera_id=camera_id,
        bbox=(x, y, x + 10, y + 10),
        timestamp=timestamp,
    )


def test_tracking_point_center_is_bbox_midpoint():
    p = _point(0, 0)
    assert p.center == (5, 5)


def test_tracking_point_distance_to_same_camera():
    a = _point(0, 0)
    b = _point(6, 8)
    assert a.distance_to(b) == pytest.approx(10.0)


def test_tracking_point_distance_across_cameras_is_infinite():
    a = _point(0, 0, camera_id="cam1")
    b = _point(0, 0, camera_id="cam2")
    assert a.distance_to(b) == float("inf")


def test_tracking_point_roundtrips_through_dict():
    p = _point(4, 8, timestamp=12.5, frame_id=3)
    restored = TrackingPoint.from_dict(p.to_dict())
    assert restored.vehicle_id == p.vehicle_id
    assert restored.bbox == p.bbox
    assert restored.timestamp == p.timestamp
    assert restored.camera_id == p.camera_id
    assert restored.center == p.center


def test_vehicle_track_keeps_points_and_trajectory():
    track = VehicleTrack("v1")
    track.add_point(_point(0, 0, timestamp=0.0))
    track.add_point(_point(10, 0, timestamp=1.0))
    assert len(track.points) == 2
    assert track.get_trajectory() == [(5, 5), (15, 5)]
    assert track.cameras_seen == {"cam1"}


def test_vehicle_track_respects_max_points():
    track = VehicleTrack("v1", max_points=2)
    for i in range(5):
        track.add_point(_point(i * 10, 0, timestamp=float(i)))
    # The deque keeps the two most recent points, so the last is i=4.
    assert len(track.points) == 2
    assert track.get_last_point().bbox[0] == 40


def test_vehicle_track_predicts_next_position():
    track = VehicleTrack("v1")
    track.add_point(_point(0, 0, timestamp=0.0))
    track.add_point(_point(10, 0, timestamp=1.0))
    # velocity is +10 px/s in x, so next center is (25, 5)
    assert track.predict_next_position("cam1") == (25, 5)


def test_vehicle_track_predicts_none_with_insufficient_points():
    track = VehicleTrack("v1")
    track.add_point(_point(0, 0, timestamp=0.0))
    assert track.predict_next_position("cam1") is None


def test_vehicle_track_roundtrips_through_dict():
    track = VehicleTrack("v1")
    track.add_point(_point(2, 4, timestamp=1.0))
    restored = VehicleTrack.from_dict(track.to_dict())
    assert restored.vehicle_id == "v1"
    assert len(restored.points) == 1


def test_speed_estimator_requires_minimum_points():
    est = SpeedEstimator(pixels_per_meter=10.0, min_track_points=5)
    track = VehicleTrack("v1")
    track.add_point(_point(0, 0, timestamp=0.0))
    track.add_point(_point(10, 0, timestamp=1.0))
    result = est.estimate_speed(track, "cam1")
    assert result["sufficient_data"] is False
    assert result["speed_mph"] == 0.0


def test_speed_estimator_converts_pixels_to_mph():
    # 100 px/s at 10 px/m = 10 m/s = 22.3694 mph
    est = SpeedEstimator(pixels_per_meter=10.0, min_track_points=2, smoothing_window=0)
    track = VehicleTrack("v1")
    for i in range(3):
        track.add_point(_point(i * 100, 0, timestamp=float(i)))
    result = est.estimate_speed(track, "cam1")
    assert result["sufficient_data"] is True
    # Reported speed is rounded to one decimal by the estimator.
    assert result["speed_mph"] == pytest.approx(22.4, abs=0.05)


def test_speed_estimator_rejects_implausible_speeds():
    est = SpeedEstimator(pixels_per_meter=1.0, min_track_points=2, max_speed_mph=150.0)
    track = VehicleTrack("v1")
    for i in range(3):
        track.add_point(_point(i * 100000, 0, timestamp=float(i)))
    result = est.estimate_speed(track, "cam1")
    assert result["sufficient_data"] is False


def test_speed_estimator_smoothing_averages_recent_samples():
    est = SpeedEstimator(pixels_per_meter=10.0, min_track_points=2, smoothing_window=3)
    track = VehicleTrack("v1")
    for i in range(4):
        track.add_point(_point(i * 100, 0, timestamp=float(i)))
    result = est.estimate_speed(track, "cam1")
    assert "speed_mph" in result
    assert result["confidence"] > 0.0


def test_calibration_sets_pixels_per_meter():
    est = SpeedEstimator()
    est.calibrate_camera(
        "cam1", [((0, 0), (0, 0)), ((100, 0), (100, 0))], reference_distance_meters=5.0
    )
    assert est._get_pixels_per_meter("cam1") == pytest.approx(20.0)


def test_default_pixels_per_meter_used_for_uncalibrated_camera():
    est = SpeedEstimator(pixels_per_meter=42.0)
    assert est._get_pixels_per_meter("unknown") == pytest.approx(42.0)


def test_multicamera_tracker_creates_and_updates_tracks():
    tracker = MultiCameraTracker()
    # A detection needs a vehicle_id, or an identifier to supply one.
    detections = [{"vehicle_id": "v1", "bbox": (0, 0, 20, 20), "confidence": 0.9}]
    tracker.update("cam1", detections, None, timestamp=0.0)
    tracks = tracker.get_active_tracks()
    assert len(tracks) == 1


def test_multicamera_tracker_skips_detections_without_an_identity():
    tracker = MultiCameraTracker()
    detections = [{"bbox": (0, 0, 20, 20), "confidence": 0.9}]
    tracker.update("cam1", detections, None, timestamp=0.0)
    assert tracker.get_active_tracks() == []


@pytest.mark.xfail(
    reason=(
        "Bug: MultiCameraTracker.camera_transitions is a defaultdict(set) but is "
        "incremented as a counter in _register_camera_transition, so any vehicle "
        "seen by a second camera raises TypeError. Tracked for a separate fix."
    ),
    strict=True,
)
def test_multicamera_tracker_records_camera_transition():
    tracker = MultiCameraTracker()
    tracker.update(
        "cam1", [{"vehicle_id": "v1", "bbox": (0, 0, 20, 20)}], None, timestamp=0.0
    )
    tracker.update(
        "cam2", [{"vehicle_id": "v1", "bbox": (0, 0, 20, 20)}], None, timestamp=1.0
    )
    assert tracker.camera_transitions[("cam1", "cam2")] == 1


def test_camera_transitions_container_is_a_set_defaultdict():
    # Pinning the container type so the fix is visible in the diff.
    from collections import defaultdict

    tracker = MultiCameraTracker()
    assert isinstance(tracker.camera_transitions, defaultdict)
    assert tracker.camera_transitions.default_factory is set


def test_load_tracks_replaces_the_container_with_an_int_defaultdict(tmp_path):
    # load_tracks rebuilds camera_transitions as defaultdict(int), so the
    # container type differs before and after a save/load cycle.
    from collections import defaultdict

    path = str(tmp_path / "tracks.json")
    tracker = MultiCameraTracker()
    tracker.save_tracks(path)
    restored = MultiCameraTracker()
    restored.load_tracks(path)
    assert isinstance(restored.camera_transitions, defaultdict)
    assert restored.camera_transitions.default_factory is int
