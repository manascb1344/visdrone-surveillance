"""Pin the behavior of metrics helpers in ``visdrone_surveillance.utils.evaluation``."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visdrone_surveillance.utils.evaluation import calculate_iou, calculate_ap


def test_iou_identical_boxes_is_one():
    box = np.array([0.0, 0.0, 10.0, 10.0])
    assert calculate_iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([20.0, 20.0, 30.0, 30.0])
    assert calculate_iou(a, b) == pytest.approx(0.0)


def test_iou_half_overlap():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([5.0, 0.0, 15.0, 10.0])
    # intersection 50, union 150
    assert calculate_iou(a, b) == pytest.approx(50.0 / 150.0)


def test_iou_zero_area_union_is_zero_not_crash():
    a = np.array([0.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 0.0])
    assert calculate_iou(a, b) == 0


def test_ap_perfect_detector_is_one():
    assert calculate_ap([0.5, 1.0], [1.0, 1.0]) == pytest.approx(1.0)


def test_ap_zero_recall_is_zero():
    assert calculate_ap([0.0], [1.0]) == pytest.approx(0.0)


def test_ap_is_bounded_by_one():
    ap = calculate_ap([0.2, 0.4, 0.6, 0.8, 1.0], [1.0, 0.9, 0.9, 0.95, 1.0])
    assert 0.0 <= ap <= 1.0


def test_ap_intermediate_value_is_stable():
    # Two detections, first is a false positive, second a true positive.
    ap = calculate_ap([0.5], [0.5, 0.5])
    assert ap == pytest.approx(0.5, abs=1e-6)
