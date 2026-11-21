"""Tests for operator selection and single-target tracking.

The sequences are synthetic: no simulator, no model, no dataset.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "main"))
sys.path.insert(0, ROOT)

from visdrone_surveillance.follow import (
    BBox,
    Detection,
    SingleTargetTracker,
    select_by_bbox,
    select_by_point,
)


def make_det(x1, y1, x2, y2, confidence=0.9, label="car"):
    return Detection(bbox=BBox(x1, y1, x2, y2), confidence=confidence, label=label)


def test_select_by_point_hits_the_box_under_the_click():
    inside = make_det(10, 10, 50, 50)
    other = make_det(200, 200, 240, 240)
    assert select_by_point([other, inside], 30, 30) is inside


def test_select_by_point_misses_return_none():
    dets = [make_det(10, 10, 50, 50)]
    assert select_by_point(dets, 500, 500) is None


def test_select_by_point_empty_input_returns_none():
    assert select_by_point([], 30, 30) is None


def test_select_by_point_smallest_box_wins_on_overlap():
    big = make_det(0, 0, 100, 100)
    small = make_det(40, 40, 60, 60)
    assert select_by_point([big, small], 50, 50) is small
    assert select_by_point([small, big], 50, 50) is small


def test_select_by_bbox_takes_the_most_confident_overlap():
    low = make_det(10, 10, 60, 60, confidence=0.5)
    high = make_det(12, 12, 62, 62, confidence=0.95)
    pick = select_by_bbox([low, high], BBox(0, 0, 80, 80))
    assert pick is high


def test_select_by_bbox_respects_min_iou():
    det = make_det(100, 100, 140, 140)
    assert select_by_bbox([det], BBox(0, 0, 80, 80), min_iou=0.3) is None


def test_select_by_bbox_empty_input_returns_none():
    assert select_by_bbox([], BBox(0, 0, 100, 100)) is None


def test_tracker_estimates_velocity_on_a_moving_track():
    tracker = SingleTargetTracker(make_det(100, 100, 140, 140), 0.0)
    for i in range(1, 6):
        x = 100 + 20 * i
        state = tracker.update([make_det(x, 100, x + 40, 140)], float(i))
    assert state.velocity[0] == pytest.approx(20.0, rel=0.2)
    assert state.velocity[1] == pytest.approx(0.0, abs=0.5)


def test_tracker_coasts_through_missed_frames_keeping_velocity():
    tracker = SingleTargetTracker(make_det(100, 100, 140, 140), 0.0)
    tracker.update([make_det(120, 100, 160, 140)], 1.0)
    tracker.update([make_det(140, 100, 180, 140)], 2.0)
    velocity = tracker.state.velocity

    state = tracker.update([], 3.0)
    assert state.missed == 1
    assert state.velocity == velocity
    assert state.bbox.center[0] > 140.0

    state = tracker.update([], 4.0)
    assert state.missed == 2
    assert state.velocity == velocity


def test_tracker_missed_counter_resets_on_match():
    tracker = SingleTargetTracker(make_det(100, 100, 140, 140), 0.0)
    tracker.update([], 1.0)
    tracker.update([], 2.0)
    assert tracker.state.missed == 2
    state = tracker.update([make_det(105, 100, 145, 140)], 3.0)
    assert state.missed == 0


def test_tracker_lost_threshold():
    tracker = SingleTargetTracker(make_det(100, 100, 140, 140), 0.0)
    tracker.update([], 1.0)
    tracker.update([], 2.0)
    assert tracker.lost(3) is False
    tracker.update([], 3.0)
    assert tracker.lost(3) is True
    assert tracker.lost(2) is True


def test_tracker_follows_the_closest_detection_not_a_distractor():
    tracker = SingleTargetTracker(make_det(100, 100, 140, 140), 0.0)
    tracker.update([make_det(104, 100, 144, 140)], 1.0)
    state = tracker.update(
        [make_det(600, 600, 640, 640), make_det(108, 100, 148, 140)], 2.0
    )
    assert state.bbox.center[0] == pytest.approx(128.0)
    assert state.missed == 0
