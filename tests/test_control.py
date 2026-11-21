"""Tests for guidance control and the follow mission state machine.

The plant is a tiny stub written in the test itself: the target sits at
a fixed world position and the camera yaw decides where it appears.
No simulator is involved.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "main"))
sys.path.insert(0, ROOT)

from visdrone_surveillance.control import (
    FollowPilot,
    GuidanceController,
    PilotState,
)
from visdrone_surveillance.follow import BBox, Detection, TargetState


def make_det(x1, y1, x2, y2, confidence=0.9):
    return Detection(bbox=BBox(x1, y1, x2, y2), confidence=confidence, label="car")


def make_state(center_x, center_y, scale, velocity=(0.0, 0.0), missed=0):
    half = scale / 2.0
    return TargetState(
        bbox=BBox(center_x - half, center_y - half, center_x + half, center_y + half),
        velocity=velocity,
        timestamp=0.0,
        missed=missed,
    )


def make_controller(**kwargs):
    params = dict(
        image_size=(640, 480),
        desired_scale=40.0,
        kp=1.0,
        ki=0.2,
        kd=0.0,
        max_forward=3.0,
        max_yaw_rate=1.0,
        max_climb=1.0,
    )
    params.update(kwargs)
    return GuidanceController(**params)


class StubPlant:
    """A target at a fixed world angle, seen through a rotating camera.

    The target world angle is zero, the camera yaw starts at some angle,
    and the image offset of the target is proportional to the yaw
    difference. Yawing right decreases the offset, matching the
    documented sign convention.
    """

    def __init__(self, width=640, height=480, offset=100.0):
        self.width = width
        self.height = height
        self.offset = offset
        self.yaw = 0.0

    def step(self, yaw_rate, dt):
        self.yaw += yaw_rate * dt
        return self.offset - self.yaw * 200.0


def test_right_offset_produces_positive_yaw_rate():
    controller = make_controller()
    command = controller.update(make_state(500.0, 240.0, 40.0), 0.05)
    assert command.yaw_rate > 0.0


def test_left_offset_produces_negative_yaw_rate():
    controller = make_controller()
    command = controller.update(make_state(140.0, 240.0, 40.0), 0.05)
    assert command.yaw_rate < 0.0


def test_offset_drives_toward_zero_in_a_closed_loop():
    controller = make_controller(kp=2.0, ki=0.0, kd=0.05)
    plant = StubPlant(offset=120.0)
    dt = 0.05
    for _ in range(200):
        offset = plant.step(0.0, dt)
        state = make_state(320.0 + offset, 240.0, 40.0)
        command = controller.update(state, dt)
        plant.step(command.yaw_rate, dt)
    assert abs(plant.offset - 120.0) < 30.0


def test_range_error_sign_both_ways():
    controller = make_controller(kp=1.0, ki=0.0, kd=0.0)

    too_far = make_state(320.0, 240.0, 20.0)
    assert controller.update(too_far, 0.05).forward > 0.0

    too_close = make_state(320.0, 240.0, 80.0)
    assert controller.update(too_close, 0.05).forward < 0.0


def test_command_is_clamped():
    controller = make_controller(
        kp=10.0, ki=0.0, kd=0.0, max_forward=2.0, max_yaw_rate=0.5, max_climb=0.4
    )
    command = controller.update(make_state(640.0, 480.0, 400.0), 0.05)
    assert command.forward <= 2.0
    assert command.yaw_rate <= 0.5
    assert command.climb <= 0.4
    assert command.forward >= -2.0
    assert command.yaw_rate >= -0.5
    assert command.climb >= -0.4


def test_integral_anti_windup_does_not_explode():
    controller = make_controller(kp=1.0, ki=5.0, kd=0.0)
    for _ in range(500):
        command = controller.update(make_state(320.0 + 200.0, 240.0, 40.0), 0.05)
    assert abs(command.yaw_rate) <= controller.max_yaw_rate

    controller.reset()
    fresh = controller.update(make_state(320.0, 240.0, 40.0), 0.05)
    assert abs(fresh.yaw_rate) < abs(command.yaw_rate)


def test_reset_clears_the_integral():
    controller = make_controller(kp=1.0, ki=5.0, kd=0.0)
    for _ in range(100):
        controller.update(make_state(520.0, 240.0, 40.0), 0.05)
    biased = controller.update(make_state(520.0, 240.0, 40.0), 0.05)
    controller.reset()
    cleared = controller.update(make_state(520.0, 240.0, 40.0), 0.05)
    assert abs(cleared.yaw_rate) < abs(biased.yaw_rate)


def test_vertical_offset_drives_climb():
    controller = make_controller()
    high = controller.update(make_state(320.0, 140.0, 40.0), 0.05)
    low = controller.update(make_state(320.0, 340.0, 40.0), 0.05)
    assert high.climb > 0.0
    assert low.climb < 0.0


def make_pilot(**kwargs):
    controller = make_controller(kp=1.0, ki=0.0, kd=0.0)
    params = dict(lock_frames=2, max_missed=3, reacquire_timeout=1.0)
    params.update(kwargs)
    return FollowPilot(controller, **params)


def test_pilot_happy_path_select_to_follow():
    pilot = make_pilot()
    assert pilot.state == PilotState.SELECT
    hover = pilot.step([make_det(100, 100, 140, 140)], 0.0, (640, 480))
    assert hover == pilot.step([], 0.0, (640, 480))
    assert hover.forward == 0.0 and hover.yaw_rate == 0.0

    pilot.set_target(make_det(100, 100, 140, 140), 0.0)
    assert pilot.state == PilotState.LOCK

    pilot.step([make_det(101, 100, 141, 140)], 0.05, (640, 480))
    assert pilot.state == PilotState.LOCK
    pilot.step([make_det(102, 100, 142, 140)], 0.10, (640, 480))
    assert pilot.state == PilotState.FOLLOW

    command = pilot.step([make_det(103, 100, 143, 140)], 0.15, (640, 480))
    assert pilot.state == PilotState.FOLLOW
    # The target is left of and above the image center, so the drone steers.
    assert command.yaw_rate != 0.0
    # The target scale equals the desired scale, so the distance hold is
    # already satisfied and no forward speed is commanded.
    assert abs(command.forward) < 1e-9


def test_pilot_forward_sign_both_ways():
    too_close = make_pilot()
    too_close.set_target(make_det(300, 220, 380, 300), 0.0)  # scale 80
    too_close.step([make_det(300, 220, 380, 300)], 0.05, (640, 480))
    too_close.step([make_det(300, 220, 380, 300)], 0.10, (640, 480))
    close_command = too_close.step([make_det(300, 220, 380, 300)], 0.15, (640, 480))
    assert close_command.forward < 0.0

    too_far = make_pilot()
    too_far.set_target(make_det(310, 230, 330, 250), 0.0)  # scale 20
    too_far.step([make_det(310, 230, 330, 250)], 0.05, (640, 480))
    too_far.step([make_det(310, 230, 330, 250)], 0.10, (640, 480))
    far_command = too_far.step([make_det(310, 230, 330, 250)], 0.15, (640, 480))
    assert far_command.forward > 0.0


def test_pilot_reacquires_and_recovers():
    pilot = make_pilot()
    pilot.set_target(make_det(100, 100, 140, 140), 0.0)
    pilot.step([make_det(100, 100, 140, 140)], 0.05, (640, 480))
    pilot.step([make_det(100, 100, 140, 140)], 0.10, (640, 480))
    assert pilot.state == PilotState.FOLLOW

    for i in range(3):
        pilot.step([], 0.15 + 0.05 * i, (640, 480))
    assert pilot.state == PilotState.REACQUIRE

    command = pilot.step([make_det(110, 100, 150, 140)], 0.35, (640, 480))
    assert pilot.state == PilotState.FOLLOW
    assert command is not None


def test_pilot_timeout_goes_lost_and_returns_zero_commands():
    pilot = make_pilot(reacquire_timeout=0.2)
    pilot.set_target(make_det(100, 100, 140, 140), 0.0)
    pilot.step([make_det(100, 100, 140, 140)], 0.05, (640, 480))
    pilot.step([make_det(100, 100, 140, 140)], 0.10, (640, 480))
    for i in range(3):
        pilot.step([], 0.15 + 0.05 * i, (640, 480))
    assert pilot.state == PilotState.REACQUIRE

    # REACQUIRE was entered at t=0.25, so the 0.2 s budget ends at t=0.45.
    pilot.step([], 0.40, (640, 480))
    assert pilot.state == PilotState.REACQUIRE

    pilot.step([], 0.50, (640, 480))
    assert pilot.state == PilotState.LOST

    command = pilot.step([make_det(100, 100, 140, 140)], 0.55, (640, 480))
    assert command.forward == 0.0
    assert command.yaw_rate == 0.0
    assert command.climb == 0.0


def test_pilot_reset_returns_to_select():
    pilot = make_pilot()
    pilot.set_target(make_det(100, 100, 140, 140), 0.0)
    pilot.step([make_det(100, 100, 140, 140)], 0.05, (640, 480))
    pilot.reset()
    assert pilot.state == PilotState.SELECT
    assert pilot.tracker is None
    command = pilot.step([make_det(100, 100, 140, 140)], 0.10, (640, 480))
    assert command.forward == 0.0


def test_pilot_is_total_across_states():
    pilot = make_pilot()
    for _ in range(5):
        pilot.step([], 0.0, (640, 480))
    pilot.set_target(make_det(100, 100, 140, 140), 0.0)
    for _ in range(20):
        pilot.step([make_det(105, 100, 145, 140)], 0.05, (640, 480))
    pilot.abort()
    assert pilot.state == PilotState.ABORT
    command = pilot.step([make_det(105, 100, 145, 140)], 1.0, (640, 480))
    assert command.forward == 0.0
