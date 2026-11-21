"""Tests for the drone follow-loop simulation.

Everything here is offline and fast: small frames, no models, no network.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from visdrone_surveillance.simulation import DroneCamera, SimScenario, World
from visdrone_surveillance.types import VelocityCommand

IMG_W = 320
IMG_H = 240


def make_camera(**kwargs) -> DroneCamera:
    """Camera with the small test frame size."""
    params = dict(
        position=(0.0, 0.0, 20.0),
        yaw=0.0,
        image_width=IMG_W,
        image_height=IMG_H,
        vfov_deg=60.0,
    )
    params.update(kwargs)
    return DroneCamera(**params)


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------
def test_world_is_deterministic_for_same_seed():
    a = World(seed=7, num_vehicles=5)
    b = World(seed=7, num_vehicles=5)
    assert [(v.x, v.y, v.heading, v.speed) for v in a.vehicles] == [
        (v.x, v.y, v.heading, v.speed) for v in b.vehicles
    ]


def test_world_step_is_deterministic():
    a = World(seed=3, num_vehicles=4)
    b = World(seed=3, num_vehicles=4)
    for _ in range(10):
        a.step(0.1)
        b.step(0.1)
    assert [(v.x, v.y, v.heading) for v in a.vehicles] == [
        (v.x, v.y, v.heading) for v in b.vehicles
    ]


def test_scenario_run_is_deterministic():
    commands = [
        VelocityCommand(forward=1.0),
        VelocityCommand(forward=1.0, yaw_rate=0.2),
        VelocityCommand(forward=0.5, climb=0.1),
    ]
    s1 = SimScenario(world=World(seed=11, num_vehicles=4), camera=make_camera())
    s2 = SimScenario(world=World(seed=11, num_vehicles=4), camera=make_camera())
    h1 = s1.run(commands, dt=0.05, seed=11)
    h2 = s2.run(commands, dt=0.05, seed=11)
    for (f1, o1, w1), (f2, o2, w2) in zip(h1, h2):
        assert np.array_equal(f1, f2)
        assert len(o1) == len(o2)
        assert [(v.x, v.y) for v in w1] == [(v.x, v.y) for v in w2]


# ----------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------
def test_projection_of_origin_is_image_center():
    cam = make_camera()
    u, v = cam.world_to_image((0.0, 0.0))
    assert u == pytest.approx(IMG_W / 2.0)
    assert v == pytest.approx(IMG_H / 2.0)


def test_projection_of_known_point():
    cam = make_camera()
    # Scale factor: focal length over altitude.
    f = (IMG_H / 2.0) / math.tan(math.radians(60.0) / 2.0)
    scale = f / 20.0
    # With yaw 0 the camera looks along +x, so +x maps to the image top.
    u, v = cam.world_to_image((5.0, 0.0))
    assert u == pytest.approx(IMG_W / 2.0)
    assert v == pytest.approx(IMG_H / 2.0 - 5.0 * scale)
    # +y is the camera right axis.
    u, v = cam.world_to_image((0.0, 5.0))
    assert u == pytest.approx(IMG_W / 2.0 + 5.0 * scale)
    assert v == pytest.approx(IMG_H / 2.0)


def test_projection_advances_point_ahead_upwards():
    cam = make_camera()
    _, v_ahead = cam.world_to_image((10.0, 0.0))
    _, v_behind = cam.world_to_image((-10.0, 0.0))
    assert v_ahead < IMG_H / 2.0
    assert v_behind > IMG_H / 2.0


def test_yaw_rotates_image_plane():
    cam = make_camera(yaw=math.pi / 2.0)
    scale = cam.pixels_per_meter
    # With a 90 degree yaw the camera looks along +y, and +x is camera left.
    u, v = cam.world_to_image((0.0, 5.0))
    assert u == pytest.approx(IMG_W / 2.0)
    assert v == pytest.approx(IMG_H / 2.0 - 5.0 * scale)
    u, v = cam.world_to_image((5.0, 0.0))
    assert u == pytest.approx(IMG_W / 2.0 - 5.0 * scale)
    assert v == pytest.approx(IMG_H / 2.0)


def test_image_to_world_round_trips():
    cam = make_camera(yaw=0.4)
    x, y = cam.image_to_world(cam.world_to_image((3.0, -2.0)))
    assert x == pytest.approx(3.0)
    assert y == pytest.approx(-2.0)


# ----------------------------------------------------------------------
# Camera integration
# ----------------------------------------------------------------------
def test_forward_command_moves_along_heading():
    cam = make_camera()
    cam.apply(VelocityCommand(forward=2.0), 0.5)
    assert cam.x == pytest.approx(1.0)
    assert cam.y == pytest.approx(0.0)


def test_yaw_rate_rotates_camera():
    cam = make_camera()
    cam.apply(VelocityCommand(yaw_rate=1.0), 0.5)
    assert cam.yaw == pytest.approx(0.5)


def test_climb_changes_altitude_and_scale():
    cam = make_camera()
    before = cam.pixels_per_meter
    cam.apply(VelocityCommand(climb=5.0), 1.0)
    assert cam.z == pytest.approx(25.0)
    assert cam.pixels_per_meter == pytest.approx(before * 20.0 / 25.0)


def test_forward_command_after_yaw_follows_new_heading():
    cam = make_camera()
    cam.apply(VelocityCommand(yaw_rate=math.pi / 2.0), 1.0)
    cam.apply(VelocityCommand(forward=1.0), 1.0)
    assert cam.x == pytest.approx(0.0, abs=1e-6)
    assert cam.y == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Rendering and observation
# ----------------------------------------------------------------------
def test_render_shape_and_dtype():
    world = World(seed=1, num_vehicles=3)
    cam = make_camera()
    frame = cam.render(world)
    assert frame.shape == (IMG_H, IMG_W, 3)
    assert frame.dtype == np.uint8


def test_render_is_deterministic():
    world = World(seed=2, num_vehicles=4)
    cam = make_camera()
    assert np.array_equal(cam.render(world), cam.render(world))


def test_observe_returns_only_in_frame_vehicles():
    world = World(seed=5, num_vehicles=4)
    cam = make_camera()
    detections = cam.observe(world)
    for det in detections:
        assert det.confidence == pytest.approx(1.0)
        assert det.label == "vehicle"
        assert det.bbox.x1 >= 0.0 and det.bbox.y1 >= 0.0
        assert det.bbox.x2 <= float(IMG_W) and det.bbox.y2 <= float(IMG_H)
        assert det.bbox.width > 0.0 and det.bbox.height > 0.0


def test_observe_excludes_distant_vehicle():
    world = World(seed=0, num_vehicles=0)
    from visdrone_surveillance.simulation.world import Vehicle

    world.vehicles.append(Vehicle(id=0, x=1000.0, y=1000.0))
    world.vehicles.append(Vehicle(id=1, x=0.0, y=0.0))
    cam = make_camera()
    detections = cam.observe(world)
    assert len(detections) == 1


def test_observation_bbox_matches_rendered_vehicle():
    world = World(seed=9, num_vehicles=0)
    from visdrone_surveillance.simulation.world import Vehicle

    world.vehicles.append(
        Vehicle(id=0, x=2.0, y=1.0, heading=0.3, length=4.5, width=1.9)
    )
    cam = make_camera()
    det = cam.observe(world)[0]
    scale = cam.pixels_per_meter
    # The bbox is the pixel hull of the rotated footprint. With camera yaw 0
    # the world x axis maps to image rows and the world y axis to image cols.
    hull_x = 4.5 * math.cos(0.3) + 1.9 * math.sin(0.3)
    hull_y = 4.5 * math.sin(0.3) + 1.9 * math.cos(0.3)
    assert det.bbox.width == pytest.approx(hull_y * scale, rel=1e-6)
    assert det.bbox.height == pytest.approx(hull_x * scale, rel=1e-6)
    frame = cam.render(world)
    cx, cy = det.bbox.center
    px = frame[int(round(cy)), int(round(cx))]
    assert tuple(int(v) for v in px) == world.vehicles[0].color


# ----------------------------------------------------------------------
# Scenario stepping
# ----------------------------------------------------------------------
def test_step_returns_frame_observations_and_states():
    scenario = SimScenario(world=World(seed=4, num_vehicles=3), camera=make_camera())
    frame, obs, states = scenario.step(VelocityCommand(forward=1.0), dt=0.05)
    assert frame.shape == (IMG_H, IMG_W, 3)
    assert isinstance(obs, list)
    assert len(states) == 3


def test_vehicles_stay_inside_world_bounds():
    scenario = SimScenario(world=World(seed=8, num_vehicles=8), camera=make_camera())
    for _ in range(200):
        scenario.step(VelocityCommand(forward=2.0, yaw_rate=0.1), dt=0.1)
        for state in scenario.world.states:
            assert scenario.world.in_bounds(state.position)


def test_run_returns_one_result_per_command():
    scenario = SimScenario(world=World(seed=6, num_vehicles=2), camera=make_camera())
    history = scenario.run([VelocityCommand(forward=1.0)] * 5, dt=0.05, seed=6)
    assert len(history) == 5
    for frame, obs, states in history:
        assert frame.shape == (IMG_H, IMG_W, 3)
        assert len(states) == 2
